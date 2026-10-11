import React, {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
} from 'react';

export interface PlaybackContextValue {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackSpeed: number;
  togglePlay: () => void;
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  restart: () => void;
  setPlaybackSpeed: (speed: number) => void;
  registerVideo: (id: string, el: HTMLVideoElement | null) => void;
  unregisterVideo: (id: string) => void;
  formatTime: (seconds: number) => string;
  isScrubbingRef: React.MutableRefObject<boolean>;
}

const PlaybackContext = createContext<PlaybackContextValue | null>(null);

export const PlaybackProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeedState] = useState<number>(1.0);

  const isScrubbingRef = useRef<boolean>(false);
  const videoElementsRef = useRef<Map<string, HTMLVideoElement>>(new Map());

  // Register / Unregister video elements
  const registerVideo = useCallback((id: string, el: HTMLVideoElement | null) => {
    if (el) {
      el.muted = true;
      el.playbackRate = playbackSpeed;
      videoElementsRef.current.set(id, el);

      // If duration is not yet set and this video has duration, adopt it
      if (el.duration && !isNaN(el.duration) && el.duration > 0) {
        setDuration(el.duration);
      }
    } else {
      videoElementsRef.current.delete(id);
    }
  }, [playbackSpeed]);

  const unregisterVideo = useCallback((id: string) => {
    videoElementsRef.current.delete(id);
  }, []);

  // Set shared playback speed
  const setPlaybackSpeed = useCallback((speed: number) => {
    setPlaybackSpeedState(speed);
    videoElementsRef.current.forEach((video) => {
      video.playbackRate = speed;
    });
  }, []);

  // Shared play
  const play = useCallback(() => {
    setIsPlaying(true);
    videoElementsRef.current.forEach((video) => {
      video.play().catch(() => {});
    });
  }, []);

  // Shared pause
  const pause = useCallback(() => {
    setIsPlaying(false);
    videoElementsRef.current.forEach((video) => {
      video.pause();
    });
  }, []);

  // Shared toggle play
  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, pause, play]);

  // Shared seek
  const seek = useCallback((time: number) => {
    setCurrentTime(time);
    videoElementsRef.current.forEach((video) => {
      video.currentTime = time;
    });
  }, []);

  // Shared restart
  const restart = useCallback(() => {
    setCurrentTime(0);
    setIsPlaying(true);
    videoElementsRef.current.forEach((video) => {
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  }, []);

  // Format time as m:ss.s
  const formatTime = useCallback((seconds: number) => {
    if (isNaN(seconds) || seconds < 0) return '0:00.0';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${m}:${s < 10 ? '0' : ''}${s}.${ms}`;
  }, []);

  // Master clock & drift correction loop using requestVideoFrameCallback
  useEffect(() => {
    let handle: number | null = null;
    let isRunning = true;

    const syncLoop = () => {
      if (!isRunning) return;

      const videos = Array.from(videoElementsRef.current.values());
      if (videos.length > 0 && !isScrubbingRef.current) {
        const master = videos[0];

        if (master.duration && !isNaN(master.duration) && master.duration > 0) {
          setDuration((prev) => (prev > 0 ? prev : master.duration));
        }

        const masterTime = master.currentTime;
        setCurrentTime(masterTime);

        // Check if master ended to trigger synchronized loop restart
        if (master.ended || (master.duration > 0 && masterTime >= master.duration - 0.05)) {
          restart();
          return;
        }

        // Correct drift (> 0.05s) on all other videos
        for (let i = 1; i < videos.length; i++) {
          const slave = videos[i];
          const diff = Math.abs(masterTime - slave.currentTime);
          if (diff > 0.05) {
            slave.currentTime = masterTime;
          }
          if (slave.playbackRate !== master.playbackRate) {
            slave.playbackRate = master.playbackRate;
          }
          if (isPlaying && slave.paused) {
            slave.play().catch(() => {});
          } else if (!isPlaying && !slave.paused) {
            slave.pause();
          }
        }
      }

      const firstVideo = videos[0];
      if (firstVideo && 'requestVideoFrameCallback' in firstVideo) {
        handle = (firstVideo as any).requestVideoFrameCallback(syncLoop);
      } else {
        handle = requestAnimationFrame(syncLoop);
      }
    };

    handle = requestAnimationFrame(syncLoop);

    return () => {
      isRunning = false;
      if (handle !== null) {
        cancelAnimationFrame(handle);
      }
    };
  }, [isPlaying, restart]);

  const value: PlaybackContextValue = {
    isPlaying,
    currentTime,
    duration,
    playbackSpeed,
    togglePlay,
    play,
    pause,
    seek,
    restart,
    setPlaybackSpeed,
    registerVideo,
    unregisterVideo,
    formatTime,
    isScrubbingRef,
  };

  return <PlaybackContext.Provider value={value}>{children}</PlaybackContext.Provider>;
};

export const usePlayback = (): PlaybackContextValue => {
  const context = useContext(PlaybackContext);
  if (!context) {
    throw new Error('usePlayback must be used within a PlaybackProvider');
  }
  return context;
};
