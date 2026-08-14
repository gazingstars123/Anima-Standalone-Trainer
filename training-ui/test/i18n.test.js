const test = require("node:test");
const assert = require("node:assert/strict");
const { createI18n } = require("../public/js/i18n.js");

function createStorage(values = {}) {
  return {
    getItem(key) {
      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : null;
    },
    setItem(key, value) {
      values[key] = String(value);
    },
  };
}

function createDocument(elements = []) {
  const listeners = new Map();
  return {
    documentElement: { lang: "" },
    querySelectorAll(selector) {
      if (selector === "[data-i18n]") {
        return elements.filter((element) => element.dataset.i18n);
      }
      if (selector === "[data-i18n-placeholder]") {
        return elements.filter((element) => element.dataset.i18nPlaceholder);
      }
      if (selector === "[data-i18n-title]") {
        return elements.filter((element) => element.dataset.i18nTitle);
      }
      return [];
    },
    dispatchEvent(event) {
      (listeners.get(event.type) || []).forEach((listener) => listener(event));
    },
    addEventListener(type, listener) {
      const current = listeners.get(type) || [];
      current.push(listener);
      listeners.set(type, current);
    },
  };
}

test("uses a saved locale before the browser locale", () => {
  const i18n = createI18n({
    storage: createStorage({ ui_locale: "en" }),
    navigator: { language: "zh-CN" },
  });

  assert.equal(i18n.getLocale(), "en");
});

test("detects Chinese browsers and falls back to English", () => {
  const chinese = createI18n({
    storage: createStorage(),
    navigator: { language: "zh-TW" },
  });
  const fallback = createI18n({
    storage: createStorage(),
    navigator: { language: "fr-FR" },
  });

  assert.equal(chinese.getLocale(), "zh-CN");
  assert.equal(fallback.getLocale(), "en");
});

test("translates known keys and interpolates parameters", () => {
  const i18n = createI18n({
    storage: createStorage({ ui_locale: "zh-CN" }),
    navigator: { language: "en-US" },
  });

  assert.equal(i18n.t("jobs.empty"), "暂无任务");
  assert.equal(i18n.t("samples.showMore", { count: 3 }), "显示全部（另外 3 个）");
});

test("keeps hardware multi-GPU translated in both locales", () => {
  const i18n = createI18n({ storage: createStorage({ ui_locale: "en" }) });

  assert.equal(i18n.t("hardware.multiGpu"), "Multi-GPU");
  i18n.setLocale("zh-CN");
  assert.equal(i18n.t("hardware.multiGpu"), "多 GPU");
});

test("falls back to the English key when a translation is missing", () => {
  const i18n = createI18n({
    storage: createStorage({ ui_locale: "zh-CN" }),
    navigator: { language: "zh-CN" },
  });

  assert.equal(i18n.t("does.not.exist"), "does.not.exist");
});

test("translates text back and forth when the locale changes", () => {
  const i18n = createI18n({
    storage: createStorage({ ui_locale: "zh-CN" }),
    navigator: { language: "zh-CN" },
  });

  assert.equal(i18n.translateText("Jobs"), "任务");
  i18n.setLocale("en");
  assert.equal(i18n.translateText("任务"), "Jobs");
});

test("translates text, placeholders, and titles without replacing form controls", () => {
  const label = {
    dataset: { i18n: "jobs.empty" },
    textContent: "",
  };
  const input = {
    dataset: { i18nPlaceholder: "prompts.placeholder" },
    placeholder: "",
  };
  const button = {
    dataset: { i18nTitle: "actions.delete" },
    title: "",
  };
  const document = createDocument([label, input, button]);
  const i18n = createI18n({
    storage: createStorage({ ui_locale: "zh-CN" }),
    navigator: { language: "en-US" },
    document,
  });

  i18n.translateDocument();

  assert.equal(label.textContent, "暂无任务");
  assert.equal(input.placeholder, "输入提示词文本...");
  assert.equal(button.title, "删除");
});
