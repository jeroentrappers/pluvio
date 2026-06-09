import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_hooks/flutter_hooks.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../../l10n/app_localizations.dart';
import '../../location/application/location_providers.dart';
import '../application/radar_providers.dart';
import '../domain/radar_animation.dart';
import 'widgets/forecast_chart.dart';
import 'widgets/precipitation_legend.dart';
import 'widgets/radar_map.dart';
import 'widgets/timeline_slider.dart';

/// One animation tick. 12 nowcast frames span ~5s — fast enough to feel
/// like motion, slow enough that each step reads.
const _playTick = Duration(milliseconds: 400);

class RadarScreen extends HookConsumerWidget {
  const RadarScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final location = ref.watch(currentLocationProvider);
    final frameIndex = useState<int?>(null);
    final isPlaying = useState(false);

    return Scaffold(
      appBar: AppBar(
        title: Text(l10n.appTitle),
        actions: [
          IconButton(
            tooltip: l10n.refresh,
            icon: const Icon(Icons.refresh),
            onPressed: () {
              final loc = location.value;
              if (loc != null) ref.invalidate(radarAnimationProvider(loc));
            },
          ),
        ],
      ),
      body: location.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (_, __) => _ErrorView(message: l10n.locationError),
        data: (latLng) => _RadarBody(
          location: latLng,
          frameIndex: frameIndex,
          isPlaying: isPlaying,
        ),
      ),
    );
  }
}

class _RadarBody extends HookConsumerWidget {
  const _RadarBody({
    required this.location,
    required this.frameIndex,
    required this.isPlaying,
  });

  final LatLng location;
  final ValueNotifier<int?> frameIndex;
  final ValueNotifier<bool> isPlaying;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final state = ref.watch(radarAnimationProvider(location));
    final anim = state.value?.valueOrNull;
    final frameCount = anim?.frames.length ?? 0;

    // Loop the animation while `isPlaying` is true. Re-arms whenever
    // play state or the underlying frame count changes.
    useEffect(() {
      if (!isPlaying.value || frameCount < 2) return null;
      final timer = Timer.periodic(_playTick, (_) {
        final current = frameIndex.value ?? anim!.currentIndex;
        frameIndex.value = (current + 1) % frameCount;
      });
      return timer.cancel;
    }, [isPlaying.value, frameCount]);

    void togglePlay() {
      // Snap back to the start of the future band when starting from idle
      // so the user sees the forecast play out, not the past loop again.
      if (!isPlaying.value && anim != null) {
        frameIndex.value = anim.currentIndex;
      }
      isPlaying.value = !isPlaying.value;
    }

    return state.when(
      loading: () => _MapShell(
        center: location,
        animation: null,
        frameIndex: 0,
        onIndexChanged: (_) {},
        isPlaying: false,
        onPlayPause: () {},
        body: const _LoadingHint(),
      ),
      error: (_, __) => _MapShell(
        center: location,
        animation: null,
        frameIndex: 0,
        onIndexChanged: (_) {},
        isPlaying: false,
        onPlayPause: () {},
        body: _ErrorView(message: l10n.radarError),
      ),
      data: (result) => result.when(
        ok: (anim) {
          final idx = frameIndex.value ?? anim.currentIndex;
          return _MapShell(
            center: location,
            animation: anim,
            frameIndex: idx,
            onIndexChanged: (v) {
              frameIndex.value = v;
              // Manual scrub pauses playback — standard video-scrubber UX.
              if (isPlaying.value) isPlaying.value = false;
            },
            isPlaying: isPlaying.value,
            onPlayPause: togglePlay,
            body: _ForecastSummary(animation: anim, currentIndex: idx),
          );
        },
        err: (_) => _MapShell(
          center: location,
          animation: null,
          frameIndex: 0,
          onIndexChanged: (_) {},
          isPlaying: false,
          onPlayPause: () {},
          body: _ErrorView(message: l10n.radarError),
        ),
      ),
    );
  }
}

class _MapShell extends StatelessWidget {
  const _MapShell({
    required this.center,
    required this.animation,
    required this.frameIndex,
    required this.onIndexChanged,
    required this.isPlaying,
    required this.onPlayPause,
    required this.body,
  });

  final LatLng center;
  final RadarAnimation? animation;
  final int frameIndex;
  final ValueChanged<int> onIndexChanged;
  final bool isPlaying;
  final VoidCallback onPlayPause;
  final Widget body;

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).padding.bottom;
    return Column(
      children: [
        Expanded(
          flex: 3,
          child: RadarMap(
            center: center,
            animation: animation,
            currentIndex: frameIndex,
          ),
        ),
        if (animation != null && !animation!.isEmpty)
          TimelineSlider(
            animation: animation!,
            currentIndex: frameIndex,
            onChanged: onIndexChanged,
            isPlaying: isPlaying,
            onPlayPause: onPlayPause,
          ),
        Expanded(
          flex: 2,
          child: Padding(
            // Extra bottom margin so the legend doesn't sit on the gesture
            // bar; respects the system safe-area where it exists.
            padding: EdgeInsets.fromLTRB(16, 8, 16, 24 + bottomInset),
            child: body,
          ),
        ),
      ],
    );
  }
}

class _ForecastSummary extends StatelessWidget {
  const _ForecastSummary({required this.animation, required this.currentIndex});

  final RadarAnimation animation;
  final int currentIndex;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final headline = _headline(l10n, animation);
    final futureFrames = animation.frames
        .where((f) => !f.timestamp.isBefore(animation.referenceTime))
        .toList();
    // Translate the global frame index into the future-only sub-list so the
    // chart highlights the right bar while the user scrubs or plays. When
    // scrubbing past observation frames there's no future bar to highlight.
    final currentFrame = animation.frames[currentIndex];
    final futureCurrent = currentFrame.timestamp.isBefore(animation.referenceTime)
        ? null
        : futureFrames.indexWhere((f) => !f.timestamp.isBefore(currentFrame.timestamp));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 12,
      children: [
        Text(headline, style: Theme.of(context).textTheme.titleLarge),
        Expanded(
          child: ForecastChart(
            frames: futureFrames,
            referenceTime: animation.referenceTime,
            currentIndex: futureCurrent,
          ),
        ),
        const PrecipitationLegend(),
      ],
    );
  }

  String _headline(AppLocalizations l10n, RadarAnimation animation) {
    final minutes = animation.minutesUntilRain;
    if (minutes == null) return l10n.nowcastDry;
    if (minutes == 0) return l10n.nowcastRaining;
    return l10n.nowcastRainInMinutes(minutes);
  }
}

class _LoadingHint extends StatelessWidget {
  const _LoadingHint();

  @override
  Widget build(BuildContext context) {
    return const Center(child: CircularProgressIndicator());
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text(message, textAlign: TextAlign.center),
      ),
    );
  }
}
