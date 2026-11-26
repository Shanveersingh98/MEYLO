import { useState, useEffect, useCallback } from 'react';
import { ThemeMode, ResolvedTheme } from '../types';

export function useTheme() {
  const [themeMode, setThemeModeState] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem('ai_companion_theme_mode');
    if (saved === 'light' || saved === 'dark' || saved === 'system') {
      return saved;
    }
    return 'dark'; // Dark mode by default as requested
  });

  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>('dark');

  const applyTheme = useCallback((mode: ThemeMode) => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    let active: ResolvedTheme = 'dark';

    if (mode === 'system') {
      active = mediaQuery.matches ? 'dark' : 'light';
    } else {
      active = mode;
    }

    setResolvedTheme(active);

    const root = document.documentElement;
    if (active === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
      root.style.colorScheme = 'dark';
    } else {
      root.classList.add('light');
      root.classList.remove('dark');
      root.style.colorScheme = 'light';
    }
  }, []);

  const setThemeMode = useCallback((mode: ThemeMode) => {
    setThemeModeState(mode);
    localStorage.setItem('ai_companion_theme_mode', mode);
    applyTheme(mode);
  }, [applyTheme]);

  const cycleTheme = useCallback(() => {
    // Cycle: dark -> light -> system -> dark
    setThemeModeState(prev => {
      let next: ThemeMode = 'dark';
      if (prev === 'dark') next = 'light';
      else if (prev === 'light') next = 'system';
      else next = 'dark';

      localStorage.setItem('ai_companion_theme_mode', next);
      applyTheme(next);
      return next;
    });
  }, [applyTheme]);

  useEffect(() => {
    applyTheme(themeMode);

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = () => {
      if (themeMode === 'system') {
        applyTheme('system');
      }
    };

    try {
      mediaQuery.addEventListener('change', handleSystemChange);
      return () => mediaQuery.removeEventListener('change', handleSystemChange);
    } catch {
      // Fallback for older browsers
      mediaQuery.addListener(handleSystemChange);
      return () => mediaQuery.removeListener(handleSystemChange);
    }
  }, [themeMode, applyTheme]);

  return {
    themeMode,
    resolvedTheme,
    setThemeMode,
    cycleTheme,
  };
}
