import './globals.css'

export const metadata = {
  title: 'STAG-INR · Subsurface Temperature Grid',
  description: 'Predict 3D subsurface ocean temperatures across the North Indian Ocean'
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  )
}
