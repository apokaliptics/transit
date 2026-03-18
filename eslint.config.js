import svelte from 'eslint-plugin-svelte';
import tsParser from '@typescript-eslint/parser';

export default [
    ...svelte.configs['flat/recommended'],
    {
        files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
        languageOptions: {
            parserOptions: {
                parser: tsParser
            }
        }
    },
    {
        ignores: [".svelte-kit/**", "src-tauri/**", "node_modules/**"]
    }
];
