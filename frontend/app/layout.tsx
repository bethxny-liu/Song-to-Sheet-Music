import { DM_Sans, Source_Serif_4 } from "next/font/google";
import { ReactNode } from "react";

import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-dm-sans"
});

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["500", "600"],
  variable: "--font-source-serif"
});

export const metadata = {
  title: "Audio to Sheet Music",
  description: "Convert audio recordings into readable sheet music"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${dmSans.variable} ${sourceSerif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
