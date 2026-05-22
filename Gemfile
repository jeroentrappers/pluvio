# Pinned for reproducible CI runs. Bump deliberately, never on autopilot.
source 'https://rubygems.org'

gem 'fastlane', '~> 2.221'

plugins_path = File.join(File.dirname(__FILE__), 'android', 'fastlane', 'Pluginfile')
eval_gemfile(plugins_path) if File.exist?(plugins_path)
