import svelte from 'eslint-plugin-svelte';

export default [
    ...svelte.configs['flat/recommended'],
    {
        ignores: ["src-tauri/**", "node_modules/**"]
    }
];
