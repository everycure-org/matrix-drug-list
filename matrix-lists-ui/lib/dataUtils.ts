import * as XLSX from 'xlsx';

export interface DataRow {
  [key: string]: string;
}

export interface DataTableState {
  data: DataRow[];
  columns: string[];
  isLoading: boolean;
  error: string | null;
}

export interface FetchOptions {
  forceRefresh?: boolean; // Keep for API compatibility but ignore
}

export async function fetchTSVData(url: string, options: FetchOptions = {}): Promise<{ data: DataRow[]; columns: string[] }> {
  try {
    console.log('Fetching data for:', url);
    
    // Use our proxy API route to avoid CORS issues
    const proxyUrl = `/api/fetch-data?url=${encodeURIComponent(url)}`;
    const response = await fetch(proxyUrl);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Failed to fetch data: ${response.statusText}`);
    }
    
    const text = await response.text();
    const lines = text.trim().split('\n');
    
    if (lines.length === 0) {
      throw new Error('Empty data file');
    }
    
    // Parse header row
    const columns = lines[0].split('\t');
    
    // Parse data rows
    const data: DataRow[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split('\t');
      const row: DataRow = {};
      
      columns.forEach((column, index) => {
        row[column] = values[index] || '';
      });
      
      data.push(row);
    }
    
    return { data, columns };
  } catch (error) {
    throw new Error(`Error fetching data: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function fetchExcelData(url: string, options: FetchOptions = {}): Promise<{ data: DataRow[]; columns: string[] }> {
  try {
    console.log('Fetching Excel data for:', url);
    
    // Use our proxy API route to avoid CORS issues
    const proxyUrl = `/api/fetch-data?url=${encodeURIComponent(url)}`;
    const response = await fetch(proxyUrl);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Failed to fetch data: ${response.statusText}`);
    }
    
    // Get the response as array buffer for Excel files
    const arrayBuffer = await response.arrayBuffer();
    
    if (arrayBuffer.byteLength === 0) {
      throw new Error('Empty Excel file');
    }
    
    // Parse Excel file
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    
    // Get first sheet
    const firstSheetName = workbook.SheetNames[0];
    if (!firstSheetName) {
      throw new Error('No sheets found in Excel file');
    }
    
    const worksheet = workbook.Sheets[firstSheetName];
    
    // Convert to JSON
    const jsonData = XLSX.utils.sheet_to_json(worksheet, { defval: '' }) as DataRow[];
    
    if (jsonData.length === 0) {
      throw new Error('No data found in Excel file');
    }
    
    // Extract columns from first row
    const columns = Object.keys(jsonData[0]);
    
    // Ensure all values are strings
    const data: DataRow[] = jsonData.map(row => {
      const stringRow: DataRow = {};
      columns.forEach(column => {
        stringRow[column] = String(row[column] || '');
      });
      return stringRow;
    });
    
    return { data, columns };
  } catch (error) {
    throw new Error(`Error fetching Excel data: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function fetchData(url: string, fileFormat: 'tsv' | 'xlsx', options: FetchOptions = {}): Promise<{ data: DataRow[]; columns: string[] }> {
  if (fileFormat === 'xlsx') {
    return fetchExcelData(url, options);
  } else {
    return fetchTSVData(url, options);
  }
}

export function filterData(data: DataRow[], searchTerm: string, filterColumns?: string[]): DataRow[] {
  if (!searchTerm.trim()) {
    return data;
  }
  
  const term = searchTerm.toLowerCase();
  return data.filter(row => {
    if (filterColumns && filterColumns.length > 0) {
      // Only search in specified columns
      return filterColumns.some(column => {
        const value = row[column];
        return value && value.toLowerCase().includes(term);
      });
    } else {
      // Fallback to searching all columns if no filterColumns specified
      return Object.values(row).some(value => 
        value.toLowerCase().includes(term)
      );
    }
  });
} 

export function getDisplayColumns(allColumns: string[], displayColumns: string[]): string[] {
  if (displayColumns.length === 0) {
    // If no display columns specified, show all columns
    return allColumns;
  }
  
  // Return only the specified display columns that exist in the data
  // This filters out any configured columns that don't actually exist in the dataset
  return displayColumns.filter(column => allColumns.includes(column));
} 