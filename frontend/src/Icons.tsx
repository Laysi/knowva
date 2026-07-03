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

// Hex-nut product mark (brand + assistant avatar).
export const BrandMark = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 3 20 7.5v9L12 21 4 16.5v-9z" />
    <circle cx="12" cy="12" r="3.4" />
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

export const AlertIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M10.3 3.9 2 18.1A2 2 0 0 0 3.7 21h16.6a2 2 0 0 0 1.7-2.9L13.7 3.9a2 2 0 0 0-3.4 0Z" />
    <path d="M12 9v4.5" />
    <path d="M12 17h.01" />
  </svg>
);

export const WrenchIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9l-3.8 3.8z" />
  </svg>
);

export const SlidersIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 7h16M4 12h16M4 17h16" />
    <circle cx="9" cy="7" r="2.1" fill="var(--color-surface)" />
    <circle cx="15" cy="12" r="2.1" fill="var(--color-surface)" />
    <circle cx="10" cy="17" r="2.1" fill="var(--color-surface)" />
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

export const ShieldIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 3.2 5 6v5.5c0 4.2 2.9 7.4 7 8.8 4.1-1.4 7-4.6 7-8.8V6z" />
    <path d="M9.3 12.2 11.2 14l3.4-3.6" />
  </svg>
);
