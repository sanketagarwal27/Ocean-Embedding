/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
    './store/**/*.{js,jsx}',
    './lib/**/*.{js,jsx}'
  ],
  theme: {
    extend: {
      colors: {
        abyss: {
          950: '#020617',
          900: '#0a1628',
          800: '#0f2138',
          700: '#16324a'
        }
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace']
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(45, 212, 191, 0.15), 0 8px 24px -8px rgba(45, 212, 191, 0.25)'
      }
    }
  },
  plugins: []
}
