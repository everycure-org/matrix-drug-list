# Matrix Lists Data Viewer

A Next.js application for viewing and searching through matrix data files with interactive tables.
This application provides a user-friendly interface to explore drug lists, disease lists,
indications, and contraindications data.

## Features

- **Multi-tab Interface**: View different data sets in organized tabs
- **Interactive Data Tables**: Sortable and paginated tables with search functionality
- **Real-time Search**: Fuzzy search across multiple columns
- **Caching**: IndexedDB caching for improved performance
- **Responsive Design**: Modern UI built with Tailwind CSS and Radix UI components
- **Data Sources**:
  - Drug Lists (TSV format)
  - Disease Lists (TSV format)
  - Indications Lists (XLSX format)
  - Contraindications Lists (XLSX format)

## Getting Started

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Technology Stack

- **Framework**: Next.js 15.4.2 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Radix UI primitives
- **Data Tables**: TanStack React Table
- **Search**: Fuse.js for fuzzy search
- **Data Processing**: XLSX library for Excel file support
- **Caching**: IndexedDB for client-side data caching

## Data Sources

The application connects to the following data sources:

- **Drugs List**: Comprehensive drug information including names, ingredients, and ATC codes
- **Diseases List**: Disease classifications with definitions and synonyms
- **Indications List**: Drug-disease indication relationships
- **Contraindications List**: Drug contraindication information

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Learn More

To learn more about the technologies used:

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Radix UI](https://www.radix-ui.com/)
- [TanStack Table](https://tanstack.com/table)
