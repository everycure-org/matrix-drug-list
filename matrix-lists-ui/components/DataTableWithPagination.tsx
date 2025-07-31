"use client"

import { useState, useMemo } from "react"
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  useReactTable,
} from "@tanstack/react-table"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { SearchInput } from "./SearchInput"
import { DataRow } from "@/lib/dataUtils"
import { searchData } from "@/lib/searchUtils"
import { getDisplayColumns } from "@/lib/dataUtils"
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react"

interface DataTableWithPaginationProps {
  data: DataRow[]
  columns: string[]
  onRowClick: (row: DataRow) => void
  isLoading?: boolean
  filterColumns?: string[]
  displayColumns?: string[]
}

export function DataTableWithPagination({ 
  data, 
  columns, 
  onRowClick, 
  isLoading = false,
  filterColumns,
  displayColumns = []
}: DataTableWithPaginationProps) {
  const [searchTerm, setSearchTerm] = useState("")

  // Determine which columns to display in the table
  const tableColumns = useMemo(() => 
    getDisplayColumns(columns, displayColumns), 
    [columns, displayColumns]
  )

  // Search and rank results with debouncing (search across all columns, not just display columns)
  const searchResults = useMemo(() => {
    // Skip search if term is too short
    if (searchTerm.trim().length < 2) {
      return data.map(row => ({
        row,
        score: 0,
        bestMatch: "",
        matchedColumn: ""
      }))
    }
    return searchData(data, searchTerm, filterColumns)
  }, [data, searchTerm, filterColumns])

  // Create column definitions for TanStack Table (only for display columns)
  const columnDefs: ColumnDef<DataRow>[] = useMemo(() => 
    tableColumns.map((column) => ({
      accessorKey: column,
      header: column,
      cell: ({ getValue }) => {
        const value = getValue() as string
        return (
          <div className="max-w-[200px] truncate">
            {String(value || "")}
          </div>
        )
      },
    })), [tableColumns])

  // Memoize the table data to prevent unnecessary re-processing
  const tableData = useMemo(() => {
    return searchResults.map(result => result.row)
  }, [searchResults])

  const table = useReactTable({
    data: tableData,
    columns: columnDefs,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: {
        pageSize: 50,
      },
    },
    // Disable automatic re-rendering on data changes
    autoResetPageIndex: false,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted-foreground">Loading data...</div>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted-foreground">No data loaded. Click &ldquo;Load&rdquo; to fetch data.</div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <SearchInput
        value={searchTerm}
        onChange={setSearchTerm}
        placeholder={`Search ${data.length} records...`}
      />
      
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => onRowClick(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={tableColumns.length} className="h-24 text-center">
                  {searchTerm ? `No results found for &ldquo;${searchTerm}&rdquo;` : "No results."}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination Controls */}
      <div className="flex items-center justify-between px-2">
        <div className="flex-1 text-sm text-muted-foreground">
          Showing {searchResults.length} of {data.length} results
          {searchTerm && ` (ranked by relevance for &ldquo;${searchTerm}&rdquo;)`}
        </div>
        <div className="flex items-center space-x-6 lg:space-x-8">
          <div className="flex items-center space-x-2">
            <p className="text-sm font-medium">Rows per page</p>
            <select
              value={table.getState().pagination.pageSize}
              onChange={(e) => {
                table.setPageSize(Number(e.target.value))
              }}
              className="h-8 w-[70px] rounded border border-input bg-background px-3 py-1 text-sm"
            >
              {[25, 50, 100, 200].map((pageSize) => (
                <option key={pageSize} value={pageSize}>
                  {pageSize}
                </option>
              ))}
            </select>
          </div>
          <div className="flex w-[100px] items-center justify-center text-sm font-medium">
            Page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              className="hidden h-8 w-8 p-0 lg:flex"
              onClick={() => table.setPageIndex(0)}
              disabled={!table.getCanPreviousPage()}
            >
              <span className="sr-only">Go to first page</span>
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              className="h-8 w-8 p-0"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
            >
              <span className="sr-only">Go to previous page</span>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              className="h-8 w-8 p-0"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
            >
              <span className="sr-only">Go to next page</span>
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              className="hidden h-8 w-8 p-0 lg:flex"
              onClick={() => table.setPageIndex(table.getPageCount() - 1)}
              disabled={!table.getCanNextPage()}
            >
              <span className="sr-only">Go to last page</span>
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
} 