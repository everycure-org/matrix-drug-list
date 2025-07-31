import Fuse from 'fuse.js'
import { DataRow } from './dataUtils'

export interface SearchResult {
  row: DataRow
  score: number
  bestMatch: string
  matchedColumn: string
}

export function searchData(data: DataRow[], searchTerm: string, filterColumns?: string[]): SearchResult[] {
  if (!searchTerm.trim()) {
    return data.map(row => ({
      row,
      score: 0,
      bestMatch: "",
      matchedColumn: ""
    }))
  }

  // Configure Fuse.js options based on filterColumns
  const fuseOptions = {
    // Search specific columns if provided, otherwise all fields
    keys: filterColumns && filterColumns.length > 0 ? filterColumns : ['*'],
    // Threshold for fuzzy matching (0.0 = perfect match, 1.0 = very loose)
    threshold: 0.4, // Slightly more strict to reduce false positives
    // Include score in results
    includeScore: true,
    // Include matches in results
    includeMatches: true,
    // Ignore location (don't penalize matches at end of string)
    ignoreLocation: true,
    // Use extended search (supports patterns like '!term' for negation)
    useExtendedSearch: false,
    // Minimum length for fuzzy matching
    minMatchCharLength: 2,
    // Distance for fuzzy matching (reduced for better performance)
    distance: 50,
    // Should the search be case sensitive?
    isCaseSensitive: false,
    // Should the search be sorted by score?
    shouldSort: true,
    // Limit results (reduced for better performance)
    limit: 500,
  }

  // Create Fuse instance directly without caching
  const fuse = new Fuse(data, fuseOptions)

  // Perform search
  const results = fuse.search(searchTerm)
  
  // Convert to our SearchResult format
  return results.map(result => {
    const bestMatch = result.matches?.[0]?.value || ""
    const matchedColumn = result.matches?.[0]?.key || ""
    
    return {
      row: result.item,
      score: result.score || 0,
      bestMatch,
      matchedColumn
    }
  })
} 