import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const url = searchParams.get('url')
  
  if (!url) {
    return NextResponse.json({ error: 'URL parameter is required' }, { status: 400 })
  }
  
  try {
    const response = await fetch(url)
    
    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to fetch data: ${response.statusText}` },
        { status: response.status }
      )
    }
    
    // Check if this is an Excel file based on URL
    const isExcelFile = url.toLowerCase().endsWith('.xlsx') || url.toLowerCase().endsWith('.xls')
    
    if (isExcelFile) {
      // Handle Excel files as binary data
      const arrayBuffer = await response.arrayBuffer()
      
      return new NextResponse(arrayBuffer, {
        headers: {
          'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          'Access-Control-Allow-Origin': '*',
        },
      })
    } else {
      // Handle TSV/text files
      const text = await response.text()
      
      return new NextResponse(text, {
        headers: {
          'Content-Type': 'text/plain',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }
  } catch (error) {
    return NextResponse.json(
      { error: `Error fetching data: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
} 