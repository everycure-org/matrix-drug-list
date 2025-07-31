"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { DataTableWithPagination } from "@/components/DataTableWithPagination"
import { DetailsDrawer } from "@/components/DetailsDrawer"
import { fetchData, DataRow, DataTableState, FetchOptions } from "@/lib/dataUtils"
import { dataFiles, DataFileConfig } from "@/lib/dataConfig"
import { indexedDBCache } from "@/lib/indexedDBCache"
import { RefreshCw } from "lucide-react"

interface TabData extends DataFileConfig {
  state: DataTableState
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
  
  // Track which tabs have been loaded to avoid reloading
  const loadedTabs = useRef<Set<number>>(new Set())

  const getTabId = (tabIndex: number): string => {
    return `tab-${tabIndex}-${dataFiles[tabIndex]?.type}`
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

  const handleRefresh = useCallback(async () => {
    try {
      console.log('Clearing all cache and refreshing current tab...')
      
      // Clear all cache
      await indexedDBCache.clear()
      
      // Reset loaded tabs tracking
      loadedTabs.current.clear()
      
      // Reload current tab with force refresh
      const currentTabIndex = parseInt(activeTab)
      await handleLoadData(currentTabIndex, true)
      
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
        <div>
          <h1 className="text-3xl font-bold">Matrix Lists Data Viewer</h1>
          <p className="text-muted-foreground mt-2">
            View and search through data files with interactive tables
          </p>
        </div>
        <Button 
          variant="outline" 
          size="sm"
          onClick={handleRefresh}
          title="Clear cache and refresh current tab"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          {tabsData.map((tab, index) => (
            <TabsTrigger key={index} value={index.toString()}>
              {tab.name}
            </TabsTrigger>
          ))}
        </TabsList>

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
      </Tabs>

      <DetailsDrawer
        isOpen={isDrawerOpen}
        onClose={handleDrawerClose}
        selectedRow={selectedRow}
        columns={selectedRow ? Object.keys(selectedRow) : []}
      />
    </div>
  )
}
