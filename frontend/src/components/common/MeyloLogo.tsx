import React from 'react';

interface MeyloLogoProps {
  className?: string;
  size?: number;
}

export const MeyloLogo: React.FC<MeyloLogoProps> = ({
  className = "w-5 h-5",
  size = 24
}) => {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      width={size}
      height={size}
    >
      <defs>
        <linearGradient id="meyloGradInner" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="50%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#06b6d4" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#meyloGradInner)" />
      <path
        d="M7 22.5V11.5l9 7.5 9-7.5v11"
        stroke="#ffffff"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="16" cy="10" r="2.2" fill="#38bdf8" />
      <circle cx="16" cy="10" r="1.1" fill="#ffffff" />
    </svg>
  );
};

export default MeyloLogo;
