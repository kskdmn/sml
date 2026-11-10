def format_elapsed(seconds: float) -> str:
    """Format elapsed time as days and HH:MM:SS, discarding fractional seconds."""
    days, remainder = divmod(int(seconds), 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
