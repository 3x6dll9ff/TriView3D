/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0a0f',
        'bg-surface': '#111118',
        'bg-elevated': '#18181f',
        'accent': '#4fffff',
        'accent-dim': 'rgba(79, 255, 255, 0.12)',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '4px',
        sm: '2px',
      },
      spacing: {
        'sidebar-collapsed': '48px',
        'sidebar-expanded': '220px',
      },
    },
  },
  plugins: [],
}
