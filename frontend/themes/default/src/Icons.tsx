// Hand-written inline SVG icons. Sized via font-size (1em) and colored via
// currentColor so they inherit from their container. No external icon deps.

type IconProps = { className?: string };

const base = {
  viewBox: "0 0 24 24",
  width: "1em",
  height: "1em",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

// Knowva monogram: a K whose stem is a double-tapered ray, whose arms are star
// spikes springing from it, and whose fourth ray breaks out to the left. The
// letterform and the burst are one shape — a plain star is too generic a figure
// to hold as a mark. The four spikes are separate paths on purpose: as subpaths
// of one path their windings would cancel and punch a hole through the junction.
export const BrandMark = ({ className }: IconProps) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    width="1em"
    height="1em"
    fill="currentColor"
    aria-hidden="true"
  >
    <path d="M9.6 .6 11.5 12 9.6 23.4 7.7 12Z" />
    <path d="M9 14.6 22.3 3.2 9 10.6Z" />
    <path d="M9 13.4 22.3 20.8 9 9.4Z" />
    <path d="M10.2 10.4 1.7 12 10.2 13.6Z" />
  </svg>
);

export const ChatIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M21 15a2 2 0 0 1-2 2H8l-4 3.5V6a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z" />
    <path d="M8.5 9.5h7M8.5 12.75h4.5" />
  </svg>
);

export const DocsIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M14.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7.5z" />
    <path d="M14.5 3v4.5H19" />
    <path d="M8.5 13h7M8.5 16.5h4.5" />
  </svg>
);

export const SendIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 19V5.5" />
    <path d="M6 11.5 12 5.5l6 6" />
  </svg>
);

export const StopIcon = ({ className }: IconProps) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    width="1em"
    height="1em"
    aria-hidden="true"
  >
    <rect x="6.5" y="6.5" width="11" height="11" rx="2.6" fill="currentColor" />
  </svg>
);

export const UploadIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M21 15v3.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V15" />
    <path d="M7.5 8.5 12 4l4.5 4.5" />
    <path d="M12 4v11" />
  </svg>
);

export const CompassIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="12" cy="12" r="8.6" />
    <path d="m15.4 8.6-1.8 5-5 1.8 1.8-5z" />
  </svg>
);

export const ListIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4.6 6.6h.01M4.6 12h.01M4.6 17.4h.01" />
    <path d="M8.6 6.6h10.8M8.6 12h10.8M8.6 17.4h7" />
  </svg>
);

export const UserIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="12" cy="8.2" r="3.7" />
    <path d="M4.9 20.3a7.1 7.1 0 0 1 14.2 0" />
  </svg>
);

export const TrashIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 7h16" />
    <path d="M9.5 7V4.8h5V7" />
    <path d="M6.5 7l.8 12a2 2 0 0 0 2 1.9h5.4a2 2 0 0 0 2-1.9l.8-12" />
    <path d="M10 11v6M14 11v6" />
  </svg>
);

export const ArrowRightIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M5 12h13" />
    <path d="M12.5 5.5 19 12l-6.5 6.5" />
  </svg>
);
