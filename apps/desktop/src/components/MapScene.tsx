import type { TimelineFrame } from "../types";

interface MapSceneProps {
  frame: TimelineFrame | null;
  mode: "2d" | "3d";
}

function lonToX(lon: number, width: number): number {
  return ((lon + 180) / 360) * width;
}

function latToY(lat: number, height: number): number {
  return ((90 - lat) / 180) * height;
}

function polygonPath(coordinates: number[][], width: number, height: number): string {
  return coordinates
    .map((point, index) => {
      const x = lonToX(point[0], width);
      const y = latToY(point[1], height);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function MapScene({ frame, mode }: MapSceneProps) {
  if (!frame) {
    return <div className="map-empty">Generate a project and scrub the timeline to view tectonic history.</div>;
  }

  if (mode === "3d") {
    return (
      <div className="globe-shell">
        <div className="globe">
          {frame.plateGeometries.slice(0, 18).map((plate, index) => {
            const center = plate.geometry.coordinates[0][0];
            const x = 50 + (center[0] / 180) * 32;
            const y = 50 - (center[1] / 90) * 32;
            return <span key={plate.plateId} className="globe-dot" style={{ left: `${x}%`, top: `${y}%`, animationDelay: `${index * 50}ms` }} />;
          })}
        </div>
      </div>
    );
  }

  const width = 960;
  const height = 480;

  return (
    <svg className="map-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="2D tectonic map view">
      <defs>
        <linearGradient id="ocean" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#0e3c52" />
          <stop offset="100%" stopColor="#123047" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="url(#ocean)" rx="14" />
      {frame.plateGeometries.map((plate) => (
        <path
          key={plate.plateId}
          d={polygonPath(plate.geometry.coordinates[0], width, height)}
          className="plate-path"
          style={{
            fill: `hsl(${(plate.plateId * 35) % 360} 38% 42% / 0.55)`
          }}
        />
      ))}
      {frame.boundaryGeometries.map((boundary) => (
        <line
          key={boundary.segmentId}
          x1={lonToX(boundary.geometry.coordinates[0][0], width)}
          y1={latToY(boundary.geometry.coordinates[0][1], height)}
          x2={lonToX(boundary.geometry.coordinates[1][0], width)}
          y2={latToY(boundary.geometry.coordinates[1][1], height)}
          className={`boundary ${boundary.boundaryType}`}
        />
      ))}
    </svg>
  );
}
