"use client"

import { useState } from "react"
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer"
import { Button } from "@/components/ui/button"
import { X, Copy, Check } from "lucide-react"
import { DataRow } from "@/lib/dataUtils"

interface DetailsDrawerProps {
  isOpen: boolean
  onClose: () => void
  selectedRow: DataRow | null
  columns: string[]
}

export function DetailsDrawer({ isOpen, onClose, selectedRow, columns }: DetailsDrawerProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null)

  if (!selectedRow) return null

  const copyToClipboard = async (text: string, fieldName: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedField(fieldName)
      setTimeout(() => setCopiedField(null), 2000) // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  return (
    <Drawer open={isOpen} onOpenChange={onClose} direction="right">
      <DrawerContent className="h-full">
        <DrawerHeader className="flex items-center justify-between">
          <DrawerTitle>Row Details</DrawerTitle>
          <DrawerClose asChild>
            <Button variant="ghost" size="sm">
              <X className="h-4 w-4" />
            </Button>
          </DrawerClose>
        </DrawerHeader>
        
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {columns.map((column) => {
              const value = selectedRow[column] || ""
              const isEmpty = !value
              
              return (
                <div key={column} className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">
                    {column}
                  </label>
                  <div className="rounded-md border bg-muted/50 p-3 relative group">
                    <p className="text-sm break-words pr-8">
                      {value || <span className="text-muted-foreground">(empty)</span>}
                    </p>
                    {!isEmpty && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => copyToClipboard(value, column)}
                        title="Copy to clipboard"
                      >
                        {copiedField === column ? (
                          <Check className="h-3 w-3 text-green-600" />
                        ) : (
                          <Copy className="h-3 w-3" />
                        )}
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  )
} 