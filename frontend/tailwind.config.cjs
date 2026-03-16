module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        panel: "var(--panel)",
        border: "var(--border)",
        accent: "var(--accent)",
        accent2: "var(--accent-2)",
        text: "var(--text)",
        muted: "var(--muted)"
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 }
        },
        slideUp: {
          "0%": { opacity: 0, transform: "translateY(8px)" },
          "100%": { opacity: 1, transform: "translateY(0)" }
        },
        pulseDots: {
          "0%, 80%, 100%": { opacity: 0.2 },
          "40%": { opacity: 1 }
        }
      },
      animation: {
        fadeIn: "fadeIn 0.4s ease",
        slideUp: "slideUp 0.35s ease",
        pulseDots: "pulseDots 1.2s infinite"
      }
    }
  },
  plugins: [require("@tailwindcss/typography")]
};
