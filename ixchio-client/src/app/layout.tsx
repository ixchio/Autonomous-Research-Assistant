import type { Metadata } from "next";
import { VT323, Space_Mono } from "next/font/google";
import "./globals.css";

const vt323 = VT323({
  variable: "--font-vt323",
  weight: "400",
  subsets: ["latin"],
  display: "swap",
});

const spaceMono = Space_Mono({
  variable: "--font-space-mono",
  weight: ["400", "700"],
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ixchio — Deep Research Engine",
  description:
    "Multi-agent autonomous research assistant powered by STORM perspectives, reflection loops, and adaptive search routing. Built for deep, comprehensive research.",
  keywords: ["AI research", "deep research", "autonomous agent", "STORM", "multi-agent"],
  openGraph: {
    title: "ixchio — Deep Research Engine",
    description: "Multi-agent autonomous research assistant with STORM perspectives and adaptive search.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${vt323.variable} ${spaceMono.variable} antialiased min-h-screen bg-[#030305] text-zinc-100 font-mono`}
      >
        {children}
      </body>
    </html>
  );
}
