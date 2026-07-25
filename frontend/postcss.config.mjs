/** @type {import('postcss-load-config').Config} */
export default {
  plugins: {
    // Resolves the `@import` of config/design-tokens.css, which lives above
    // this package so the Streamlit dashboard reads the same file.
    "postcss-import": {},
    tailwindcss: {},
    autoprefixer: {},
  },
};
