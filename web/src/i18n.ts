import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import en from './locales/en.json'
import nl from './locales/nl.json'
import fr from './locales/fr.json'
import de from './locales/de.json'

// English is the source language; Dutch, French and German are translations.
export const LANGS = [
  { code: 'en', label: 'English' },
  { code: 'nl', label: 'Nederlands' },
  { code: 'fr', label: 'Français' },
  { code: 'de', label: 'Deutsch' },
] as const

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      nl: { translation: nl },
      fr: { translation: fr },
      de: { translation: de },
    },
    fallbackLng: 'en',
    supportedLngs: ['en', 'nl', 'fr', 'de'],
    nonExplicitSupportedLngs: true, // nl-BE -> nl, fr-BE -> fr, de-DE -> de
    interpolation: { escapeValue: false },
    detection: { order: ['localStorage', 'navigator'], caches: ['localStorage'] },
  })

export default i18n
