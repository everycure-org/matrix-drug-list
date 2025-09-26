// "use client"

// import { useState, useCallback, useEffect, useRef } from "react"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
// import { Button } from "@/components/ui/button"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
// import { Input } from "@/components/ui/input"
// import { DataTableWithPagination } from "@/components/DataTableWithPagination"
// import { DetailsDrawer } from "@/components/DetailsDrawer"
// import { fetchData, DataRow, DataTableState, FetchOptions } from "@/lib/dataUtils"
// import { dataFiles, DataFileConfig } from "@/lib/dataConfig"
// import { indexedDBCache } from "@/lib/indexedDBCache"
// import { RefreshCw, Download, Search } from "lucide-react"
// import Image from "next/image"

// interface TabData extends DataFileConfig {
//   state: DataTableState
// }

// interface SearchResult {
//   source: string
//   data: DataRow
//   matchedField: string
//   matchedValue: string
// }

// export default function Home() {
//   const [tabsData, setTabsData] = useState<TabData[]>(() =>
//     dataFiles.map(file => ({
//       ...file,
//       state: { data: [], columns: [], isLoading: false, error: null }
//     }))
//   )

//   const [selectedRow, setSelectedRow] = useState<DataRow | null>(null)
//   const [isDrawerOpen, setIsDrawerOpen] = useState(false)
//   const [activeTab, setActiveTab] = useState("0")
//   const [searchQuery, setSearchQuery] = useState("")
//   const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  
//   // Track which tabs have been loaded to avoid reloading
//   const loadedTabs = useRef<Set<number>>(new Set())

//   const getTabId = (tabIndex: number): string => {
//     return `tab-${tabIndex}-${dataFiles[tabIndex]?.type}`
//   }

//   // Download handler
//   const handleDownload = async (tabIndex: number) => {
//     const tab = tabsData[tabIndex]
//     const tabConfig = dataFiles[tabIndex]
    
//     // If data not loaded, load it first
//     if (!tab.state.data.length) {
//       await handleLoadData(tabIndex)
//       // Wait for state to update
//       setTimeout(() => {
//         const updatedTab = tabsData[tabIndex]
//         if (updatedTab.state.data.length) {
//           downloadData(updatedTab.state.data, tabConfig.name)
//         }
//       }, 100)
//     } else {
//       downloadData(tab.state.data, tabConfig.name)
//     }
//   }

//   const downloadData = (data: DataRow[], filename: string) => {
//     const csv = convertToCSV(data)
//     const blob = new Blob([csv], { type: 'text/csv' })
//     const url = window.URL.createObjectURL(blob)
//     const a = document.createElement('a')
//     a.setAttribute('hidden', '')
//     a.setAttribute('href', url)
//     a.setAttribute('download', `${filename}.csv`)
//     document.body.appendChild(a)
//     a.click()
//     document.body.removeChild(a)
//   }

//   const convertToCSV = (data: DataRow[]) => {
//     if (!data.length) return ''
//     const headers = Object.keys(data[0])
//     const csvHeaders = headers.join(',')
//     const csvRows = data.map(row => 
//       headers.map(header => {
//         const value = row[header]
//         // Escape quotes and wrap in quotes if contains comma
//         const stringValue = String(value || '')
//         return stringValue.includes(',') || stringValue.includes('"') 
//           ? `"${stringValue.replace(/"/g, '""')}"` 
//           : stringValue
//       }).join(',')
//     )
//     return [csvHeaders, ...csvRows].join('\n')
//   }

//   const handleLoadData = useCallback(async (tabIndex: number, forceRefresh: boolean = false) => {
//     const tabId = getTabId(tabIndex)
    
//     setTabsData(prev => {
//       const tab = prev[tabIndex]
//       if (!tab?.url) {
//         console.error("No URL configured for this tab")
//         return prev
//       }
      
//       return prev.map((t, i) => 
//         i === tabIndex 
//           ? { ...t, state: { ...t.state, isLoading: true, error: null } }
//           : t
//       )
//     })

//     try {
//       const currentTab = dataFiles[tabIndex]
//       if (!currentTab?.url) {
//         throw new Error("No URL configured for this tab")
//       }

//       let data: DataRow[]
//       let columns: string[]

//       // Check cache first unless force refresh
//       if (!forceRefresh) {
//         console.log(`Checking cache for tab: ${tabId}`)
//         const cachedData = await indexedDBCache.get(tabId)
        
//         if (cachedData) {
//           console.log(`Using cached data for tab: ${tabId}`)
//           data = cachedData.data
//           columns = cachedData.columns
//         } else {
//           console.log(`No cache found, fetching fresh data for tab: ${tabId}`)
//           const fetchedData = await fetchData(currentTab.url, currentTab.fileFormat)
//           data = fetchedData.data
//           columns = fetchedData.columns
          
//           // Cache the fetched data
//           await indexedDBCache.set(tabId, currentTab.url, data, columns)
//           console.log(`Cached data for tab: ${tabId}`)
//         }
//       } else {
//         console.log(`Force refresh: fetching fresh data for tab: ${tabId}`)
//         const fetchedData = await fetchData(currentTab.url, currentTab.fileFormat)
//         data = fetchedData.data
//         columns = fetchedData.columns
        
//         // Update cache with fresh data
//         await indexedDBCache.set(tabId, currentTab.url, data, columns)
//         console.log(`Updated cache for tab: ${tabId}`)
//       }

//       setTabsData(prev => prev.map((t, i) => 
//         i === tabIndex 
//           ? { ...t, state: { data, columns, isLoading: false, error: null } }
//           : t
//       ))
      
//       // Mark this tab as loaded
//       loadedTabs.current.add(tabIndex)
//     } catch (error) {
//       setTabsData(prev => prev.map((t, i) => 
//         i === tabIndex 
//           ? { ...t, state: { ...t.state, isLoading: false, error: error instanceof Error ? error.message : 'Unknown error' } }
//           : t
//       ))
//     }
//   }, [])

//   // Load all data on mount for downloads and search
//   useEffect(() => {
//     dataFiles.forEach((_, index) => {
//       if (!loadedTabs.current.has(index)) {
//         handleLoadData(index)
//       }
//     })
//   }, [handleLoadData])

//   const handleRefresh = useCallback(async () => {
//     try {
//       console.log('Clearing all cache and refreshing current tab...')
      
//       // Clear all cache
//       await indexedDBCache.clear()
      
//       // Reset loaded tabs tracking
//       loadedTabs.current.clear()
      
//       // Reload current tab with force refresh
//       const currentTabIndex = parseInt(activeTab)
//       if (currentTabIndex < dataFiles.length) {
//         await handleLoadData(currentTabIndex, true)
//       }
      
//       console.log('Cache cleared and current tab refreshed')
//     } catch (error) {
//       console.error('Failed to refresh:', error)
//     }
//   }, [activeTab, handleLoadData])

//   // Auto-load data when tab is selected
//   useEffect(() => {
//     const tabIndex = parseInt(activeTab)
//     const tabConfig = dataFiles[tabIndex]
    
//     // Load data if this tab hasn't been loaded yet and has a URL
//     if (tabConfig && 
//         tabConfig.url && 
//         !loadedTabs.current.has(tabIndex)) {
//       handleLoadData(tabIndex)
//     }
//   }, [activeTab, handleLoadData])

//   // Fuzzy search function
//   const fuzzySearch = (text: string, query: string): boolean => {
//     const textLower = text.toLowerCase()
//     const queryLower = query.toLowerCase()
    
//     // First check for exact substring match
//     if (textLower.includes(queryLower)) return true
    
//     // Fuzzy search: allow characters to be out of order
//     let queryIndex = 0
//     for (let i = 0; i < textLower.length && queryIndex < queryLower.length; i++) {
//       if (textLower[i] === queryLower[queryIndex]) {
//         queryIndex++
//       }
//     }
//     return queryIndex === queryLower.length
//   }

//   // Search across all datasets
//   const performSearch = useCallback((query: string) => {
//     if (!query.trim()) {
//       setSearchResults([])
//       return
//     }

//     const results: SearchResult[] = []
//     const seenRows = new Set<string>()

//     tabsData.forEach((tab, tabIndex) => {
//       if (tab.state.data.length === 0) return

//       tab.state.data.forEach((row, rowIndex) => {
//         // Create a unique key for this row
//         const rowKey = `${tabIndex}-${rowIndex}`
        
//         // Skip if we've already added this row
//         if (seenRows.has(rowKey)) return

//         // Check if any field in this row matches the query
//         let matchFound = false
//         let matchedField = ''
//         let matchedValue = ''
//         let bestScore = -1

//         Object.entries(row).forEach(([key, value]) => {
//           if (value && fuzzySearch(String(value), query)) {
//             const valueStr = String(value)
//             // Prioritize exact matches over fuzzy matches
//             const isExactMatch = valueStr.toLowerCase().includes(query.toLowerCase())
//             const currentScore = isExactMatch ? 1 : 0
            
//             if (!matchFound || currentScore > bestScore) {
//               matchFound = true
//               matchedField = key
//               matchedValue = valueStr
//               bestScore = currentScore
//             }
//           }
//         })

//         if (matchFound) {
//           seenRows.add(rowKey)
//           results.push({
//             source: tab.name,
//             data: row,
//             matchedField: matchedField,
//             matchedValue: matchedValue
//           })
//         }
//       })
//     })

//     // Sort results to prioritize exact matches
//     results.sort((a, b) => {
//       const aExact = a.matchedValue.toLowerCase().includes(query.toLowerCase()) ? 1 : 0
//       const bExact = b.matchedValue.toLowerCase().includes(query.toLowerCase()) ? 1 : 0
//       return bExact - aExact
//     })

//     setSearchResults(results.slice(0, 50)) // Limit to first 50 results
//   }, [tabsData])

//   useEffect(() => {
//     const delayDebounceFn = setTimeout(() => {
//       if (activeTab === "all") {
//         performSearch(searchQuery)
//       }
//     }, 900)

//     return () => clearTimeout(delayDebounceFn)
//   }, [searchQuery, activeTab, performSearch])

//   const handleRowClick = (row: DataRow) => {
//     setSelectedRow(row)
//     setIsDrawerOpen(true)
//   }

//   const handleDrawerClose = () => {
//     setIsDrawerOpen(false)
//     setSelectedRow(null)
//   }

//   return (
//     <div className="container mx-auto p-6">
//       <div className="mb-6 flex items-start justify-between">
//         <div className="flex-1">
//           <h1 className="text-3xl font-bold"> Medicines, Diseases, and Indications (MeDI): A Foundational Resource for Drug Repurposing</h1>
//           <p className="text-muted-foreground mt-2">
//             The Medicines, Diseases, and Indications (MeDI) database is a collection of tabulated data sources for drug repurposing
//           </p>
          
//           {/* Download buttons */}
//           <div className="flex flex-wrap gap-2 mt-4">
//             {dataFiles.map((file, index) => (
//               <Button
//                 key={index}
//                 variant="outline"
//                 size="sm"
//                 onClick={() => handleDownload(index)}
//                 disabled={tabsData[index].state.isLoading}
//               >
//                 <Download className="h-4 w-4 mr-2" />
//                 Download {file.name}
//               </Button>
//             ))}
//           </div>
//         </div>
        
//         <div className="flex items-center gap-4">
//           <Button 
//             variant="outline" 
//             size="sm"
//             onClick={handleRefresh}
//             title="Clear cache and refresh current tab"
//           >
//             <RefreshCw className="h-4 w-4 mr-2" />
//             Refresh
//           </Button>
          
//           {/* Logo */}
//           <a href="https://everycure.org" target="_blank" rel="noopener noreferrer">
//             <Image 
//               src="/logo.png" 
//               alt="EveryCure Logo" 
//               width={200} 
//               height={100}
//               className="cursor-pointer"
//             />
//           </a>
//         </div>
//       </div>

//       <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
//         <TabsList>
//           {tabsData.map((tab, index) => (
//             <TabsTrigger key={index} value={index.toString()}>
//               {tab.name}
//             </TabsTrigger>
//           ))}
//           <TabsTrigger value="all">All of MeDI</TabsTrigger>
//         </TabsList>

//         {/* Individual tab contents */}
//         {tabsData.map((tab, index) => (
//           <TabsContent key={index} value={index.toString()} className="space-y-4">
//             <div className="flex items-center justify-between">
//               <div className="space-y-1">
//                 <h2 className="text-xl font-semibold">{tab.name}</h2>
//                 <div className="flex items-center gap-4 text-sm text-muted-foreground">
//                   <span>Type: {tab.type}</span>
//                   <span>Format: {tab.fileFormat.toUpperCase()}</span>
//                   {tab.state.data.length > 0 && (
//                     <span>{tab.state.data.length} records</span>
//                   )}
//                 </div>
//                 {tab.state.error && (
//                   <p className="text-red-500 text-sm">{tab.state.error}</p>
//                 )}
//               </div>
//             </div>

//             <DataTableWithPagination
//               data={tab.state.data}
//               columns={tab.state.columns}
//               onRowClick={handleRowClick}
//               isLoading={tab.state.isLoading}
//               filterColumns={tab.filterColumns}
//               displayColumns={tab.displayColumns}
//             />
//           </TabsContent>
//         ))}

//         {/* All of MeDI tab */}
//         <TabsContent value="all" className="space-y-4">
//           <div className="space-y-4">
//             <div className="space-y-1">
//               <h2 className="text-xl font-semibold">All of MeDI</h2>
//               <p className="text-sm text-muted-foreground">
//                 Search across all MeDI assets
//               </p>
//             </div>

//             {/* Search input */}
//             <div className="relative">
//               <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
//               <Input
//                 placeholder="Search drugs, diseases, indications, or contraindications..."
//                 value={searchQuery}
//                 onChange={(e) => setSearchQuery(e.target.value)}
//                 className="pl-8"
//               />
//             </div>

//             {/* Search results */}
//             {searchQuery && (
//               <div className="space-y-2">
//                 {searchResults.length === 0 ? (
//                   <p className="text-muted-foreground text-center py-8">
//                     No results found for "{searchQuery}"
//                   </p>
//                 ) : (
//                   <>
//                     <p className="text-sm text-muted-foreground">
//                       Found {searchResults.length} results
//                     </p>
//                     <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
//                       {searchResults.map((result, index) => (
//                         <Card 
//                           key={index} 
//                           className="cursor-pointer hover:shadow-lg transition-shadow"
//                           onClick={() => handleRowClick(result.data)}
//                         >
//                           <CardHeader className="pb-3">
//                             <div className="flex items-center justify-between">
//                               <CardTitle className="text-sm font-medium">
//                                 {result.source}
//                               </CardTitle>
//                               <span className="text-xs text-muted-foreground">
//                                 {result.matchedField}
//                               </span>
//                             </div>
//                           </CardHeader>
//                           <CardContent>
//                             <div className="space-y-2">
//                               {Object.entries(result.data).slice(0, 3).map(([key, value]) => (
//                                 <div key={key} className="text-sm">
//                                   <span className="font-medium">{key}:</span>{" "}
//                                   <span className="text-muted-foreground">
//                                     {String(value || 'N/A').substring(0, 50)}
//                                     {String(value || '').length > 50 && '...'}
//                                   </span>
//                                 </div>
//                               ))}
//                             </div>
//                           </CardContent>
//                         </Card>
//                       ))}
//                     </div>
//                   </>
//                 )}
//               </div>
//             )}
//           </div>
//         </TabsContent>
//       </Tabs>

//       <DetailsDrawer
//         isOpen={isDrawerOpen}
//         onClose={handleDrawerClose}
//         selectedRow={selectedRow}
//         columns={selectedRow ? Object.keys(selectedRow) : []}
//       />

//       {/* Funding acknowledgment */}
//       <div className="mt-12 pt-6 border-t text-center text-sm text-muted-foreground">
//         This work was funded by the Advanced Research Projects Agency for Health (ARPA-H). MeDI was built in collaboration with the University of North Carolina and the Renaissance Computing Institute
//       </div>
//     </div>
//   )
// }












































"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { DataTableWithPagination } from "@/components/DataTableWithPagination"
import { DetailsDrawer } from "@/components/DetailsDrawer"
import { fetchData, DataRow, DataTableState, FetchOptions } from "@/lib/dataUtils"
import { dataFiles, DataFileConfig } from "@/lib/dataConfig"
import { indexedDBCache } from "@/lib/indexedDBCache"
import { RefreshCw, Download, Search } from "lucide-react"
import Image from "next/image"

interface TabData extends DataFileConfig {
  state: DataTableState
}

interface SearchResult {
  source: string
  data: DataRow
  matchedField: string
  matchedValue: string
}

// Helper function to get card display information based on source type
const getCardDisplayInfo = (result: SearchResult) => {
  const source = result.source.toLowerCase()
  const data = result.data
  
  if (source.includes('drug')) {
    // For drugs: title = drug label, subtitle = "Drug List"
    const title = data.label || data.drug_name || data.name || 'Unknown Drug'
    return {
      title,
      subtitle: 'Drug List',
      subtitleColor: 'text-muted-foreground',
      displayFields: [
        { key: 'primary_id', label: 'Primary ID', value: data.primary_id },
        { key: 'regions_of_approval', label: 'Regions of Approval', value: data.regions_of_approval },
        { key: 'label', label: 'Label', value: data.label }
      ].filter(field => field.value) // Only show fields that have values
    }
  } else if (source.includes('disease')) {
    // For diseases: title = disease name, subtitle = "Disease List"
    const title = data.disease_name || data.label || data.name || 'Unknown Disease'
    return {
      title,
      subtitle: 'Disease List',
      subtitleColor: 'text-muted-foreground',
      displayFields: [
        { key: 'disease_id', label: 'Disease ID', value: data.disease_id },
        { key: 'disease_name', label: 'Disease Name', value: data.disease_name },
        { key: 'label', label: 'Label', value: data.label }
      ].filter(field => field.value)
    }
  } else if (source.includes('indication')) {
    // For indications: title = drug -> disease, subtitle = "Indication" in green
    const drugName = data.drug_name || data.drug_label || 'Unknown Drug'
    const diseaseName = data.disease_name || data.disease_label || 'Unknown Disease'
    const title = `${drugName} → ${diseaseName}`
    return {
      title,
      subtitle: 'Indication',
      subtitleColor: 'text-green-600',
      displayFields: [
        { key: 'drug_name', label: 'Drug', value: data.drug_name },
        { key: 'disease_name', label: 'Disease', value: data.disease_name },
        { key: 'indication_type', label: 'Type', value: data.indication_type }
      ].filter(field => field.value)
    }
  } else if (source.includes('contraindication')) {
    // For contraindications: title = drug -> disease, subtitle = "Contraindication" in red
    const drugName = data.drug_name || data.drug_label || 'Unknown Drug'
    const diseaseName = data.disease_name || data.disease_label || 'Unknown Disease'
    const title = `${drugName} → ${diseaseName}`
    return {
      title,
      subtitle: 'Contraindication',
      subtitleColor: 'text-red-600',
      displayFields: [
        { key: 'drug_name', label: 'Drug', value: data.drug_name },
        { key: 'disease_name', label: 'Disease', value: data.disease_name },
        { key: 'contraindication_type', label: 'Type', value: data.contraindication_type }
      ].filter(field => field.value)
    }
  } else {
    // Default fallback
    const firstValue = Object.values(data)[0]
    return {
      title: String(firstValue || 'Unknown'),
      subtitle: result.source,
      subtitleColor: 'text-muted-foreground',
      displayFields: Object.entries(data).slice(0, 3).map(([key, value]) => ({
        key,
        label: key,
        value
      }))
    }
  }
}

export default function Home() {
  const [tabsData, setTabsData] = useState<TabData[]>(() =>
    dataFiles.map(file => ({
      ...file,
      state: { data: [], columns: [], isLoading: false, error: null }
    }))
  )

  const [selectedRow, setSelectedRow] = useState<DataRow | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("0")
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  
  // Track which tabs have been loaded to avoid reloading
  const loadedTabs = useRef<Set<number>>(new Set())

  const getTabId = (tabIndex: number): string => {
    return `tab-${tabIndex}-${dataFiles[tabIndex]?.type}`
  }

  // Download handler
  const handleDownload = async (tabIndex: number) => {
    const tab = tabsData[tabIndex]
    const tabConfig = dataFiles[tabIndex]
    
    // If data not loaded, load it first
    if (!tab.state.data.length) {
      await handleLoadData(tabIndex)
      // Wait for state to update
      setTimeout(() => {
        const updatedTab = tabsData[tabIndex]
        if (updatedTab.state.data.length) {
          downloadData(updatedTab.state.data, tabConfig.name)
        }
      }, 100)
    } else {
      downloadData(tab.state.data, tabConfig.name)
    }
  }

  const downloadData = (data: DataRow[], filename: string) => {
    const csv = convertToCSV(data)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.setAttribute('hidden', '')
    a.setAttribute('href', url)
    a.setAttribute('download', `${filename}.csv`)
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const convertToCSV = (data: DataRow[]) => {
    if (!data.length) return ''
    const headers = Object.keys(data[0])
    const csvHeaders = headers.join(',')
    const csvRows = data.map(row => 
      headers.map(header => {
        const value = row[header]
        // Escape quotes and wrap in quotes if contains comma
        const stringValue = String(value || '')
        return stringValue.includes(',') || stringValue.includes('"') 
          ? `"${stringValue.replace(/"/g, '""')}"` 
          : stringValue
      }).join(',')
    )
    return [csvHeaders, ...csvRows].join('\n')
  }

  const handleLoadData = useCallback(async (tabIndex: number, forceRefresh: boolean = false) => {
    const tabId = getTabId(tabIndex)
    
    setTabsData(prev => {
      const tab = prev[tabIndex]
      if (!tab?.url) {
        console.error("No URL configured for this tab")
        return prev
      }
      
      return prev.map((t, i) => 
        i === tabIndex 
          ? { ...t, state: { ...t.state, isLoading: true, error: null } }
          : t
      )
    })

    try {
      const currentTab = dataFiles[tabIndex]
      if (!currentTab?.url) {
        throw new Error("No URL configured for this tab")
      }

      let data: DataRow[]
      let columns: string[]

      // Check cache first unless force refresh
      if (!forceRefresh) {
        console.log(`Checking cache for tab: ${tabId}`)
        const cachedData = await indexedDBCache.get(tabId)
        
        if (cachedData) {
          console.log(`Using cached data for tab: ${tabId}`)
          data = cachedData.data
          columns = cachedData.columns
        } else {
          console.log(`No cache found, fetching fresh data for tab: ${tabId}`)
          const fetchedData = await fetchData(currentTab.url, currentTab.fileFormat)
          data = fetchedData.data
          columns = fetchedData.columns
          
          // Cache the fetched data
          await indexedDBCache.set(tabId, currentTab.url, data, columns)
          console.log(`Cached data for tab: ${tabId}`)
        }
      } else {
        console.log(`Force refresh: fetching fresh data for tab: ${tabId}`)
        const fetchedData = await fetchData(currentTab.url, currentTab.fileFormat)
        data = fetchedData.data
        columns = fetchedData.columns
        
        // Update cache with fresh data
        await indexedDBCache.set(tabId, currentTab.url, data, columns)
        console.log(`Updated cache for tab: ${tabId}`)
      }

      setTabsData(prev => prev.map((t, i) => 
        i === tabIndex 
          ? { ...t, state: { data, columns, isLoading: false, error: null } }
          : t
      ))
      
      // Mark this tab as loaded
      loadedTabs.current.add(tabIndex)
    } catch (error) {
      setTabsData(prev => prev.map((t, i) => 
        i === tabIndex 
          ? { ...t, state: { ...t.state, isLoading: false, error: error instanceof Error ? error.message : 'Unknown error' } }
          : t
      ))
    }
  }, [])

  // Load all data on mount for downloads and search
  useEffect(() => {
    dataFiles.forEach((_, index) => {
      if (!loadedTabs.current.has(index)) {
        handleLoadData(index)
      }
    })
  }, [handleLoadData])

  const handleRefresh = useCallback(async () => {
    try {
      console.log('Clearing all cache and refreshing current tab...')
      
      // Clear all cache
      await indexedDBCache.clear()
      
      // Reset loaded tabs tracking
      loadedTabs.current.clear()
      
      // Reload current tab with force refresh
      const currentTabIndex = parseInt(activeTab)
      if (currentTabIndex < dataFiles.length) {
        await handleLoadData(currentTabIndex, true)
      }
      
      console.log('Cache cleared and current tab refreshed')
    } catch (error) {
      console.error('Failed to refresh:', error)
    }
  }, [activeTab, handleLoadData])

  // Auto-load data when tab is selected
  useEffect(() => {
    const tabIndex = parseInt(activeTab)
    const tabConfig = dataFiles[tabIndex]
    
    // Load data if this tab hasn't been loaded yet and has a URL
    if (tabConfig && 
        tabConfig.url && 
        !loadedTabs.current.has(tabIndex)) {
      handleLoadData(tabIndex)
    }
  }, [activeTab, handleLoadData])

  // Fuzzy search function
  const fuzzySearch = (text: string, query: string): boolean => {
    const textLower = text.toLowerCase()
    const queryLower = query.toLowerCase()
    
    // First check for exact substring match
    if (textLower.includes(queryLower)) return true
    
    // Fuzzy search: allow characters to be out of order
    let queryIndex = 0
    for (let i = 0; i < textLower.length && queryIndex < queryLower.length; i++) {
      if (textLower[i] === queryLower[queryIndex]) {
        queryIndex++
      }
    }
    return queryIndex === queryLower.length
  }

  // Search across all datasets
  const performSearch = useCallback((query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    const results: SearchResult[] = []
    const seenRows = new Set<string>()

    tabsData.forEach((tab, tabIndex) => {
      if (tab.state.data.length === 0) return

      tab.state.data.forEach((row, rowIndex) => {
        // Create a unique key for this row
        const rowKey = `${tabIndex}-${rowIndex}`
        
        // Skip if we've already added this row
        if (seenRows.has(rowKey)) return

        // Check if any field in this row matches the query
        let matchFound = false
        let matchedField = ''
        let matchedValue = ''
        let bestScore = -1

        Object.entries(row).forEach(([key, value]) => {
          if (value && fuzzySearch(String(value), query)) {
            const valueStr = String(value)
            // Prioritize exact matches over fuzzy matches
            const isExactMatch = valueStr.toLowerCase().includes(query.toLowerCase())
            const currentScore = isExactMatch ? 1 : 0
            
            if (!matchFound || currentScore > bestScore) {
              matchFound = true
              matchedField = key
              matchedValue = valueStr
              bestScore = currentScore
            }
          }
        })

        if (matchFound) {
          seenRows.add(rowKey)
          results.push({
            source: tab.name,
            data: row,
            matchedField: matchedField,
            matchedValue: matchedValue
          })
        }
      })
    })

    // Sort results to prioritize exact matches
    results.sort((a, b) => {
      const aExact = a.matchedValue.toLowerCase().includes(query.toLowerCase()) ? 1 : 0
      const bExact = b.matchedValue.toLowerCase().includes(query.toLowerCase()) ? 1 : 0
      return bExact - aExact
    })

    setSearchResults(results.slice(0, 50)) // Limit to first 50 results
  }, [tabsData])

  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      if (activeTab === "all") {
        performSearch(searchQuery)
      }
    }, 900)

    return () => clearTimeout(delayDebounceFn)
  }, [searchQuery, activeTab, performSearch])

  const handleRowClick = (row: DataRow) => {
    setSelectedRow(row)
    setIsDrawerOpen(true)
  }

  const handleDrawerClose = () => {
    setIsDrawerOpen(false)
    setSelectedRow(null)
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6 flex items-start justify-between">
        <div className="flex-1">
          <h1 className="text-3xl font-bold"> Medicines, Diseases, and Indications (MeDI): A Foundational Resource for Drug Repurposing</h1>
          <p className="text-muted-foreground mt-2">
            The Medicines, Diseases, and Indications (MeDI) database is a collection of tabulated data sources for drug repurposing
          </p>
          
          {/* Download buttons */}
          <div className="flex flex-wrap gap-2 mt-4">
            {dataFiles.map((file, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleDownload(index)}
                disabled={tabsData[index].state.isLoading}
              >
                <Download className="h-4 w-4 mr-2" />
                Download {file.name}
              </Button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleRefresh}
            title="Clear cache and refresh current tab"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          
          {/* Logo */}
          <a href="https://everycure.org" target="_blank" rel="noopener noreferrer">
            <Image 
              src="/logo.png" 
              alt="EveryCure Logo" 
              width={200} 
              height={100}
              className="cursor-pointer"
            />
          </a>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          {tabsData.map((tab, index) => (
            <TabsTrigger key={index} value={index.toString()}>
              {tab.name}
            </TabsTrigger>
          ))}
          <TabsTrigger value="all">All of MeDI</TabsTrigger>
        </TabsList>

        {/* Individual tab contents */}
        {tabsData.map((tab, index) => (
          <TabsContent key={index} value={index.toString()} className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h2 className="text-xl font-semibold">{tab.name}</h2>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>Type: {tab.type}</span>
                  <span>Format: {tab.fileFormat.toUpperCase()}</span>
                  {tab.state.data.length > 0 && (
                    <span>{tab.state.data.length} records</span>
                  )}
                </div>
                {tab.state.error && (
                  <p className="text-red-500 text-sm">{tab.state.error}</p>
                )}
              </div>
            </div>

            <DataTableWithPagination
              data={tab.state.data}
              columns={tab.state.columns}
              onRowClick={handleRowClick}
              isLoading={tab.state.isLoading}
              filterColumns={tab.filterColumns}
              displayColumns={tab.displayColumns}
            />
          </TabsContent>
        ))}

        {/* All of MeDI tab */}
        <TabsContent value="all" className="space-y-4">
          <div className="space-y-4">
            <div className="space-y-1">
              <h2 className="text-xl font-semibold">All of MeDI</h2>
              <p className="text-sm text-muted-foreground">
                Search across all MeDI assets
              </p>
            </div>

            {/* Search input */}
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search drugs, diseases, indications, or contraindications..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
            </div>

            {/* Search results */}
            {searchQuery && (
              <div className="space-y-2">
                {searchResults.length === 0 ? (
                  <p className="text-muted-foreground text-center py-8">
                    No results found for "{searchQuery}"
                  </p>
                ) : (
                  <>
                    <p className="text-sm text-muted-foreground">
                      Found {searchResults.length} results
                    </p>
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                      {searchResults.map((result, index) => {
                        const cardInfo = getCardDisplayInfo(result)
                        return (
                          <Card 
                            key={index} 
                            className="cursor-pointer hover:shadow-lg transition-shadow"
                            onClick={() => handleRowClick(result.data)}
                          >
                            <CardHeader className="pb-3">
                              <div className="flex items-start justify-between">
                                <CardTitle className="text-sm font-medium leading-tight flex-1 mr-2">
                                  {cardInfo.title}
                                </CardTitle>
                                <span className={`text-xs font-medium whitespace-nowrap ${cardInfo.subtitleColor}`}>
                                  {cardInfo.subtitle}
                                </span>
                              </div>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-2">
                                {cardInfo.displayFields.slice(0, 3).map((field) => (
                                  <div key={field.key} className="text-sm">
                                    <span className="font-medium">{field.label}:</span>{" "}
                                    <span className="text-muted-foreground">
                                      {String(field.value || 'N/A').substring(0, 50)}
                                      {String(field.value || '').length > 50 && '...'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </CardContent>
                          </Card>
                        )
                      })}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>

      <DetailsDrawer
        isOpen={isDrawerOpen}
        onClose={handleDrawerClose}
        selectedRow={selectedRow}
        columns={selectedRow ? Object.keys(selectedRow) : []}
      />

      {/* Funding acknowledgment */}
      <div className="mt-12 pt-6 border-t text-center text-sm text-muted-foreground">
        This work was funded by the Advanced Research Projects Agency for Health (ARPA-H). MeDI was built in collaboration with the University of North Carolina and the Renaissance Computing Institute
      </div>
    </div>
  )
}