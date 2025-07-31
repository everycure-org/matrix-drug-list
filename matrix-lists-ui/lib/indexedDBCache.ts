import { DataRow } from './dataUtils'

interface CachedTabData {
  id: string
  data: DataRow[]
  columns: string[]
  timestamp: number
  url: string
}

class IndexedDBCache {
  private dbName = 'matrix-lists-cache'
  private dbVersion = 1
  private storeName = 'tabs'
  private db: IDBDatabase | null = null

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion)
      
      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        this.db = request.result
        resolve()
      }
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' })
          store.createIndex('url', 'url', { unique: false })
          store.createIndex('timestamp', 'timestamp', { unique: false })
        }
      }
    })
  }

  private async ensureDB(): Promise<IDBDatabase> {
    if (!this.db) {
      await this.init()
    }
    if (!this.db) {
      throw new Error('Failed to initialize IndexedDB')
    }
    return this.db
  }

  async get(tabId: string): Promise<{ data: DataRow[]; columns: string[] } | null> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readonly')
        const store = transaction.objectStore(this.storeName)
        const request = store.get(tabId)
        
        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
          const result = request.result as CachedTabData | undefined
          if (result) {
            resolve({ data: result.data, columns: result.columns })
          } else {
            resolve(null)
          }
        }
      })
    } catch (error) {
      console.warn('Failed to get cached data:', error)
      return null
    }
  }

  async set(tabId: string, url: string, data: DataRow[], columns: string[]): Promise<void> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readwrite')
        const store = transaction.objectStore(this.storeName)
        
        const cachedData: CachedTabData = {
          id: tabId,
          data,
          columns,
          timestamp: Date.now(),
          url
        }
        
        const request = store.put(cachedData)
        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })
    } catch (error) {
      console.warn('Failed to cache data:', error)
    }
  }

  async has(tabId: string): Promise<boolean> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readonly')
        const store = transaction.objectStore(this.storeName)
        const request = store.getKey(tabId)
        
        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
          resolve(request.result !== undefined)
        }
      })
    } catch (error) {
      console.warn('Failed to check cached data:', error)
      return false
    }
  }

  async clear(): Promise<void> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readwrite')
        const store = transaction.objectStore(this.storeName)
        const request = store.clear()
        
        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })
    } catch (error) {
      console.warn('Failed to clear cache:', error)
    }
  }

  async delete(tabId: string): Promise<void> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readwrite')
        const store = transaction.objectStore(this.storeName)
        const request = store.delete(tabId)
        
        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })
    } catch (error) {
      console.warn('Failed to delete cached data:', error)
    }
  }

  async getAllCachedTabs(): Promise<string[]> {
    try {
      const db = await this.ensureDB()
      return new Promise((resolve, reject) => {
        const transaction = db.transaction([this.storeName], 'readonly')
        const store = transaction.objectStore(this.storeName)
        const request = store.getAllKeys()
        
        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
          resolve(request.result as string[])
        }
      })
    } catch (error) {
      console.warn('Failed to get cached tab list:', error)
      return []
    }
  }
}

export const indexedDBCache = new IndexedDBCache() 