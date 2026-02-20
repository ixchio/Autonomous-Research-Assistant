import type { Metadata } from "next";
import { VT323, Space_Mono } from "next/font/google";
import "./globals.css";

const vt323 = VT323({
  variable: "--font-vt323",
  weight: "400",
  subsets: ["latin"],
});

const spaceMono = Space_Mono({
  variable: "--font-space-mono",
  weight: ["400", "700"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ixchio Deep Research",
  description: "YC-Quality Autonomous Research Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${vt323.variable} ${spaceMono.variable} antialiased min-h-screen bg-black text-zinc-100 font-mono`}
      >
        {children}
      </body>
    </html>
  );
}
