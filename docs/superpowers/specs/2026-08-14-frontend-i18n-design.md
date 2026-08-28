# Frontend Internationalization Design

## Goal

Add Chinese and English UI support to the standalone training frontend without changing training configuration values or user-entered prompt content.

## Behavior

- Supported locales are `en` and `zh-CN`.
- A saved `localStorage.ui_locale` value takes precedence over the browser locale.
- Without a saved value, `navigator.language` values beginning with `zh` select Chinese; all others select English.
- The language control is placed in the sidebar footer above Global Settings.
- Changing language updates visible labels immediately and persists the selection.
- Technical identifiers, model names, optimizer values, paths, logs, and user prompts remain unchanged.

## Architecture

`training-ui/public/js/i18n.js` owns locale resolution, translation lookup, interpolation, DOM translation, and the `localechange` event. It exposes a browser API as `window.animaI18n` and a CommonJS factory for Node tests.

Static text is translated through exact text mappings and optional `data-i18n`, `data-i18n-placeholder`, and `data-i18n-title` attributes. Dynamic strings in `app.js` use the `tr()` helper so hardware monitoring, dialogs, toasts, samples, prompts, TensorBoard, and global settings react to locale changes.

## Data Safety

Language state is stored only in browser local storage. No language value is sent to the training API or written into job configuration. Locale changes do not rebuild or reset training configuration values.

## Verification

- Node unit tests cover locale precedence, browser fallback, interpolation, missing-key fallback, DOM attributes, and switching back from Chinese to English.
- `node --check` validates both browser scripts.
- Manual browser checks cover first-load detection, persistence after reload, modal text, dynamic hardware text, and the sidebar layout.
