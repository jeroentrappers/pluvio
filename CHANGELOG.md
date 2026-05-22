# Changelog

All notable changes to Pluvio are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold: Flutter 3.44, fvm-pinned.
- Layered radar feature: domain / data / application / presentation.
- KMI WMS radar source + per-location nowcast source.
- Brussels-centred map with animated radar timeline.
- Riverpod state, Dio HTTP with retry + error mapping, Talker logging.
- Localization in NL, FR, DE, EN.
- Test pyramid: domain, DTO, source, repository, widget tests.
- Fastlane lanes for Android (verify, build_aab, deploy_internal, deploy_production) and iOS (verify, build_ipa, deploy_testflight, deploy_production).
- GitHub Actions CI: format, analyze, test, build APK/iOS debug.
