import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Matrix Lists Data Viewer",
  description: "View and search through data files with interactive tables",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
