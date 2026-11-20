import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    // CI fails on a warning too (`npm run lint` passes --max-warnings 0),
    // so a rule is either on or off, never a pile nobody reads.
    linterOptions: {
      reportUnusedDisableDirectives: 'error',
    },
    rules: {
      // eslint-plugin-react-hooks 7 ships the React Compiler's rules in its
      // recommended set. The app does not use the compiler (vite.config.ts
      // runs a plain react()), so these judge the code against an optimiser
      // that never runs: "latest value" refs read in render, state reset in
      // an effect when a dialog opens, Date.now() in render to re-check on
      // each poll. Turn them back on together with the compiler, not before.
      'react-hooks/set-state-in-effect': 'off',
      'react-hooks/refs': 'off',
      'react-hooks/immutability': 'off',
      'react-hooks/purity': 'off',
      'react-hooks/incompatible-library': 'off',
      // Only protects hot reload in dev: a file that exports a helper next
      // to its component reloads the page instead of patching it. Not worth
      // splitting files for.
      'react-refresh/only-export-components': 'off',
      // `const { type: _t, ...rest } = x` is how a key is dropped.
      '@typescript-eslint/no-unused-vars': ['error', { ignoreRestSiblings: true }],
    },
  },
])
