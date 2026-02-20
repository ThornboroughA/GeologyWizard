import { useMemo, useState } from "react";

import { _GlobeView as GlobeView, MapView } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";

import type { TimelineFrameRender } from "../types";

interface MapSceneProps {
  frame: TimelineFrameRender | null;
  mode: "2d" | "3d";
  overlay:
    | "none"
    | "velocity"
    | "boundary_class"
    | "event_confidence"
    | "uplift"
    | "subsidence"
    | "boundary_state"
    | "crust_age"
    | "craton"
    | "orogeny_phase"
    | "subduction_flux";
}

type ViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
};

function plateFillColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const derived = String(properties.derived ?? "");
  if (overlay === "none" && derived === "surface_mask") {
    const relief = Number(properties.relief ?? 0.5);
    const normalized = Math.max(0, Math.min(1, (relief - 0.25) / 0.55));
    return [
      Math.round(72 + normalized * 110),
      Math.round(108 + normalized * 90),
      Math.round(70 + normalized * 60),
      182
    ];
  }

  const plateId = Number(properties.plateId ?? 0);
  if (overlay === "velocity") {
    const velocity = Number(properties.velocityCmYr ?? 0);
    const normalized = Math.max(0, Math.min(1, velocity / 14));
    const r = Math.round(70 + normalized * 160);
    const g = Math.round(120 + normalized * 80);
    const b = Math.round(210 - normalized * 110);
    return [r, g, b, 185];
  }

  if (overlay === "event_confidence") {
    const continuity = Number(properties.continuityScore ?? 0);
    const normalized = Math.max(0, Math.min(1, continuity));
    const r = Math.round(200 - normalized * 70);
    const g = Math.round(110 + normalized * 100);
    const b = Math.round(60 + normalized * 40);
    return [r, g, b, 180];
  }

  if (overlay === "crust_age") {
    const oceanicAgeMeanMyr = Number(properties.oceanicAgeMeanMyr ?? properties.oceanicAgeP99Myr ?? 0);
    const normalized = Math.max(0, Math.min(1, oceanicAgeMeanMyr / 320));
    const r = Math.round(90 + normalized * 120);
    const g = Math.round(170 - normalized * 95);
    const b = Math.round(205 - normalized * 110);
    return [r, g, b, 185];
  }

  if (overlay === "uplift") {
    const uplift = Number(properties.upliftMean ?? properties.tectonicPotentialMean ?? 0);
    const normalized = Math.max(0, Math.min(1, uplift));
    return [Math.round(170 + normalized * 70), Math.round(120 + normalized * 85), 92, 190];
  }

  if (overlay === "subsidence") {
    const subsidence = Number(properties.subsidenceMean ?? 0);
    const normalized = Math.max(0, Math.min(1, subsidence));
    return [84, Math.round(120 + normalized * 72), Math.round(170 + normalized * 70), 188];
  }

  const base = (plateId * 37) % 255;
  return [70 + (base % 90), 105 + ((base * 2) % 85), 120 + ((base * 3) % 80), 170];
}

function boundaryColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const boundaryType = String(properties.boundaryType ?? "transform");
  if (overlay === "uplift") {
    return boundaryType === "convergent" ? [255, 204, 120, 245] : [210, 220, 228, 110];
  }
  if (overlay === "subsidence") {
    return boundaryType === "divergent" ? [140, 255, 212, 245] : [210, 220, 228, 110];
  }
  if (overlay === "boundary_state") {
    const stateClass = String(properties.stateClass ?? "passive_margin");
    if (stateClass === "subduction") {
      return [255, 174, 104, 245];
    }
    if (stateClass === "collision" || stateClass === "suture") {
      return [255, 120, 120, 245];
    }
    if (stateClass === "ridge" || stateClass === "rift") {
      return [120, 240, 255, 230];
    }
    if (stateClass === "transform") {
      return [232, 218, 255, 225];
    }
    return [205, 218, 225, 150];
  }
  if (overlay === "subduction_flux") {
    const flux = Number(properties.subductionFlux ?? 0);
    const normalized = Math.max(0, Math.min(1, flux / 20));
    return [Math.round(180 + normalized * 75), Math.round(210 - normalized * 90), 120, 210];
  }

  if (boundaryType === "convergent") {
    return [255, 208, 128, 235];
  }
  if (boundaryType === "divergent") {
    return [145, 255, 220, 230];
  }
  return [246, 234, 217, 205];
}

function overlayColor(properties: Record<string, unknown>, overlay: MapSceneProps["overlay"]): [number, number, number, number] {
  const confidence = Number(properties.confidence ?? 0);
  if (overlay === "orogeny_phase") {
    const phase = String(properties.phase ?? "active");
    if (phase === "initiation") {
      return [255, 226, 152, 165];
    }
    if (phase === "decay") {
      return [218, 186, 142, 130];
    }
    return [255, 152, 102, 190];
  }
  if (overlay === "event_confidence") {
    const alpha = Math.round(75 + Math.max(0, Math.min(1, confidence)) * 170);
    return [250, 234, 170, alpha];
  }
  return [0, 0, 0, 0];
}

export function MapScene({ frame, mode, overlay }: MapSceneProps) {
  const [viewState2d, setViewState2d] = useState<ViewState>({
    longitude: 0,
    latitude: 15,
    zoom: 0.7,
    pitch: 0,
    bearing: 0
  });
  const [viewState3d, setViewState3d] = useState<ViewState>({
    longitude: 0,
    latitude: 12,
    zoom: 0.15,
    pitch: 20,
    bearing: 0
  });

  const effectiveFrame = frame ?? {
    timeMa: 0,
    source: "cache" as const,
    nearestTimeMa: 0,
    landmassGeoJson: { type: "FeatureCollection" as const, features: [] },
    continentGeoJson: { type: "FeatureCollection" as const, features: [] },
    cratonGeoJson: { type: "FeatureCollection" as const, features: [] },
    boundaryGeoJson: { type: "FeatureCollection" as const, features: [] },
    overlayGeoJson: { type: "FeatureCollection" as const, features: [] },
    coastlineGeoJson: { type: "FeatureCollection" as const, features: [] },
    activeBeltsGeoJson: { type: "FeatureCollection" as const, features: [] },
    reliefFieldRef: null
  };

  const layers = useMemo(
    () => {
      const landData =
        effectiveFrame.continentGeoJson && effectiveFrame.continentGeoJson.features.length > 0
          ? effectiveFrame.continentGeoJson
          : effectiveFrame.landmassGeoJson;
      const boundaryData =
        effectiveFrame.activeBeltsGeoJson && effectiveFrame.activeBeltsGeoJson.features.length > 0
          ? effectiveFrame.activeBeltsGeoJson
          : effectiveFrame.boundaryGeoJson;

      return [
      new GeoJsonLayer({
        id: "landmasses",
        data: landData,
        filled: true,
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: 1,
        getFillColor: (feature: { properties: Record<string, unknown> }) => plateFillColor(feature.properties, overlay),
        getLineColor: [225, 236, 244, 146]
      }),
      new GeoJsonLayer({
        id: "coastlines",
        data: effectiveFrame.coastlineGeoJson ?? { type: "FeatureCollection", features: [] },
        filled: false,
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: 1.2,
        getLineColor: [218, 238, 248, 160]
      }),
      new GeoJsonLayer({
        id: "cratons",
        data: effectiveFrame.cratonGeoJson ?? { type: "FeatureCollection", features: [] },
        filled: overlay === "craton",
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: overlay === "craton" ? 2.0 : 1.4,
        getFillColor: [248, 233, 162, 65],
        getLineColor: overlay === "craton" ? [255, 226, 128, 245] : [248, 233, 162, 190]
      }),
      new GeoJsonLayer({
        id: "boundaries",
        data: boundaryData,
        filled: false,
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: overlay === "uplift" || overlay === "subsidence" ? 2.8 : 1.8,
        getLineColor: (feature: { properties: Record<string, unknown> }) => boundaryColor(feature.properties, overlay)
      }),
      new GeoJsonLayer({
        id: "events",
        data: effectiveFrame.overlayGeoJson,
        filled: overlay === "event_confidence" || overlay === "orogeny_phase",
        stroked: true,
        wrapLongitude: true,
        lineWidthMinPixels: overlay === "event_confidence" || overlay === "orogeny_phase" ? 1.6 : 0.8,
        getFillColor: (feature: { properties: Record<string, unknown> }) => overlayColor(feature.properties, overlay),
        getLineColor: (feature: { properties: Record<string, unknown> }) => overlayColor(feature.properties, overlay)
      })
      ];
    },
    [effectiveFrame, overlay]
  );

  if (!frame) {
    return <div className="map-empty">Generate a project and scrub the timeline to view tectonic history.</div>;
  }

  const view = mode === "2d" ? new MapView({ repeat: true }) : new GlobeView({});
  const viewState = mode === "2d" ? viewState2d : viewState3d;

  return (
    <div className={`deck-shell ${mode === "3d" ? "deck-shell-globe" : ""}`}>
      <DeckGL
        views={view}
        viewState={viewState}
        controller
        onViewStateChange={({ viewState: nextViewState }) => {
          const casted = nextViewState as ViewState;
          if (mode === "2d") {
            setViewState2d(casted);
          } else {
            setViewState3d(casted);
          }
        }}
        layers={layers}
      />
    </div>
  );
}
