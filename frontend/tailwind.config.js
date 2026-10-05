/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
          // Teal as text/icon colour; lighter in dark (see index.css).
          ink: "var(--primary-ink)",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
          ink: "var(--destructive-ink)",
          subtle: "var(--destructive-subtle)",
          border: "var(--destructive-border)",
        },
        warning: {
          ink: "var(--warning-ink)",
          subtle: "var(--warning-subtle)",
          border: "var(--warning-border)",
        },
        info: {
          ink: "var(--info-ink)",
          subtle: "var(--info-subtle)",
          border: "var(--info-border)",
        },
        success: {
          ink: "var(--success-ink)",
          subtle: "var(--success-subtle)",
          border: "var(--success-border)",
        },
        "good-ink": "var(--good-ink)",
        "bad-ink": "var(--bad-ink)",
        middle: "var(--middle)",
        "chart-grid": "var(--chart-grid)",
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
