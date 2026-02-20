import { useMemo } from "react";

interface TimelineScrubberProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

export function TimelineScrubber({ value, min, max, step, onChange }: TimelineScrubberProps) {
  const percent = useMemo(() => ((max - value) / (max - min)) * 100, [max, min, value]);

  return (
    <section className="timeline-card">
      <div className="timeline-header">
        <h2>Geologic Timeline</h2>
        <span>{value} Ma</span>
      </div>
      <input
        className="timeline-input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label="Geologic timeline scrubber"
      />
      <div className="timeline-scale" style={{ "--progress": `${percent}%` } as React.CSSProperties}>
        <span>{max} Ma</span>
        <span>{Math.floor((max + min) / 2)} Ma</span>
        <span>{min} Ma</span>
      </div>
    </section>
  );
}
