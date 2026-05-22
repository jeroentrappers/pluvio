import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../../domain/radar_animation.dart';

/// Base map with a single radar overlay rendered from a flutter_map [TileLayer].
/// We rebuild the overlay layer whenever the active frame changes — that's
/// cheap because tiles for the new TIME are fetched in the background while
/// the previous frame stays visible.
class RadarMap extends StatelessWidget {
  const RadarMap({
    super.key,
    required this.center,
    required this.animation,
    required this.currentIndex,
  });

  final LatLng center;
  final RadarAnimation? animation;
  final int currentIndex;

  @override
  Widget build(BuildContext context) {
    final frame = (animation == null || animation!.isEmpty)
        ? null
        : animation!.frames[currentIndex];

    return FlutterMap(
      options: MapOptions(
        initialCenter: center,
        initialZoom: 8,
        minZoom: 5,
        maxZoom: 13,
        interactionOptions: const InteractionOptions(
          flags: InteractiveFlag.pinchZoom |
              InteractiveFlag.drag |
              InteractiveFlag.doubleTapZoom,
        ),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'app.appmire.pluvio',
          maxNativeZoom: 19,
        ),
        if (frame != null)
          _RadarOverlay(frame: frame),
        const _AttributionBox(),
      ],
    );
  }
}

class _RadarOverlay extends StatelessWidget {
  const _RadarOverlay({required this.frame});

  final RadarFrame frame;

  @override
  Widget build(BuildContext context) {
    return TileLayer(
      // Using WMS rasters: we pass through the same URL for every tile and
      // let flutter_map's tile coordinator handle z/x/y under the hood by
      // computing BBOX. A future iteration can switch to a real WMS layer
      // (`wmsOptions`) once we've validated the layer name with KMI.
      urlTemplate: frame.tileUrlTemplate,
      tileProvider: NetworkTileProvider(),
      key: ValueKey(frame.timestamp),
    );
  }
}

class _AttributionBox extends StatelessWidget {
  const _AttributionBox();

  @override
  Widget build(BuildContext context) {
    return const RichAttributionWidget(
      attributions: [
        TextSourceAttribution('© OpenStreetMap contributors'),
        TextSourceAttribution('Radar © KMI / IRM (opendata.meteo.be)'),
      ],
    );
  }
}
