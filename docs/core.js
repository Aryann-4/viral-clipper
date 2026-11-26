const TERMINAL_RE = /[.!?]["')\]]?\s*$/;
const HOOK_RE = /\b(why|how|secret|mistake|most|never|stop|start|first|real|truth|nobody|everything)\b/i;
const KEYWORDS = [
  "because", "result", "proven", "best", "worst", "easy", "fast", "free",
  "before", "after", "but", "however", "secret", "simple", "actual",
];

export function normalizeWords(chunks) {
  const words = [];
  for (const chunk of chunks || []) {
    const text = String(chunk?.text || "").trim();
    const timestamp = chunk?.timestamp;
    if (!text || !Array.isArray(timestamp) || timestamp.length < 2) continue;
    const start = Number(timestamp[0]);
    let end = Number(timestamp[1]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    end = Math.min(end, start + 12);
    words.push({ start, end, text });
  }
  return words;
}

export function sentenceUnits(words) {
  const sentences = [];
  let current = [];
  for (const word of words) {
    if (current.length) {
      const last = current[current.length - 1];
      const gap = word.start - last.end;
      const span = last.end - current[0].start;
      if (gap > 0.7 || TERMINAL_RE.test(last.text) || span > 9) {
        sentences.push(current);
        current = [];
      }
    }
    current.push(word);
  }
  if (current.length) sentences.push(current);

  return sentences.map((group, index) => ({
    start: group[0].start,
    end: group[group.length - 1].end,
    text: group.map((word) => word.text).join(" ").replace(/\s+([,.!?])/g, "$1"),
    words: group,
    index,
  }));
}

export function generateCandidates(sentences, targetDuration, totalDuration) {
  if (!sentences.length) return [];
  const minDuration = Math.max(8, targetDuration * 0.78);
  const maxDuration = targetDuration * 1.22;
  const hardMax = Math.min(totalDuration, maxDuration + 2);
  const candidates = [];
  const seen = new Set();

  for (let i = 0; i < sentences.length; i += 1) {
    if (sentences[i].start >= totalDuration - 0.5) break;
    let best = null;
    for (let j = i; j < sentences.length; j += 1) {
      const end = Math.min(totalDuration, sentences[j].end);
      const duration = end - sentences[i].start;
      if (duration > hardMax) break;
      if (duration >= minDuration && duration <= hardMax) {
        const distance = Math.abs(duration - targetDuration);
        if (!best || distance < best.distance) best = { distance, j, end };
      }
    }
    if (!best) continue;
    const key = `${i}:${best.j}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const words = sentences.slice(i, best.j + 1).flatMap((sentence) => sentence.words);
    const text = words.map((word) => word.text).join(" ").replace(/\s+([,.!?])/g, "$1");
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    if (wordCount < 5) continue;
    const start = Math.max(0, sentences[i].start);
    if (best.end - start < 2) continue;
    candidates.push({
      start,
      end: best.end,
      duration: best.end - start,
      text,
      wordCount,
      sentenceStart: i,
      sentenceEnd: best.j,
    });
  }

  if (!candidates.length) {
    const fallbackMax = Math.min(totalDuration, Math.max(maxDuration, targetDuration * 1.35));
    const start = Math.max(0, sentences[0].start);
    const end = Math.min(totalDuration, start + fallbackMax);
    const words = sentences
      .flatMap((sentence) => sentence.words)
      .filter((word) => word.end <= end + 0.01);
    const text = words.map((word) => word.text).join(" ").replace(/\s+([,.!?])/g, "$1");
    if (end - start >= 2 && text.split(/\s+/).filter(Boolean).length >= 5) {
      candidates.push({
        start,
        end,
        duration: end - start,
        text,
        wordCount: text.split(/\s+/).filter(Boolean).length,
        sentenceStart: 0,
        sentenceEnd: Math.max(0, sentences.findIndex((sentence) => sentence.start >= end) - 1),
      });
    }
  }
  return candidates;
}

export function energyEnvelope(samples, sampleRate, hopSeconds = 0.5) {
  if (!samples || !samples.length || !sampleRate) return null;
  const hop = Math.max(1, Math.floor(sampleRate * hopSeconds));
  const values = [];
  for (let i = 0; i < samples.length; i += hop) {
    let sum = 0;
    let count = 0;
    for (let j = i; j < Math.min(samples.length, i + hop); j += 1) {
      sum += samples[j] * samples[j];
      count += 1;
    }
    const rms = count ? Math.sqrt(sum / count) : 0;
    values.push(20 * Math.log10(rms + 1e-8));
  }
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return {
    db: values,
    hopSeconds,
    p10: sorted[Math.floor(sorted.length * 0.1)],
    p90: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))],
  };
}

export function scoreInterval(envelope, start, end) {
  if (!envelope || !envelope.db.length || end <= start) return 0;
  const i0 = Math.max(0, Math.floor(start / envelope.hopSeconds));
  const i1 = Math.min(envelope.db.length, Math.ceil(end / envelope.hopSeconds));
  const segment = envelope.db.slice(i0, Math.max(i0 + 1, i1));
  if (!segment.length) return 0;
  const mean = segment.reduce((sum, value) => sum + value, 0) / segment.length;
  const span = Math.max(4, envelope.p90 - envelope.p10);
  return Math.max(0, Math.min(1, (mean - envelope.p10) / span));
}

const clip = (value) => Math.max(0, Math.min(1, value));

export function scoreCandidates(candidates, envelope, targetDuration) {
  for (const candidate of candidates) {
    const words = candidate.text.toLowerCase();
    const firstWords = words.split(/\s+/).slice(0, 12).join(" ");
    const hook = clip((HOOK_RE.test(firstWords) ? 0.75 : 0.15) + (/\?/.test(firstWords) ? 0.25 : 0));
    const energy = scoreInterval(envelope, candidate.start, candidate.end);
    const keywordHits = KEYWORDS.reduce((count, word) => {
      return count + (words.includes(word) ? 1 : 0);
    }, 0);
    const keywords = clip(keywordHits / 3);
    const wpm = (candidate.wordCount / candidate.duration) * 60;
    const pace = clip(1 - Math.abs(wpm - 155) / 90);
    const lastWord = words.trim().split(/\s+/).pop() || "";
    const completeness = TERMINAL_RE.test(lastWord) ? 1 : 0.55;
    const length = clip(1 - Math.abs(candidate.duration - targetDuration) / targetDuration);
    candidate.wpm = Math.round(wpm);
    candidate.breakdown = { hook, energy, keywords, pace, completeness, length };
    candidate.score = Math.round(
      100 * (
        0.24 * hook +
        0.20 * energy +
        0.16 * keywords +
        0.14 * pace +
        0.12 * completeness +
        0.14 * length
      ) * 10
    ) / 10;
  }
  return candidates;
}

export function selectClips(candidates, count) {
  const picked = [];
  for (const candidate of [...candidates].sort((a, b) => b.score - a.score)) {
    if (picked.length >= count) break;
    const overlaps = picked.some((item) => {
      return candidate.start < item.end + 0.75 && item.start < candidate.end + 0.75;
    });
    if (!overlaps) picked.push(candidate);
  }
  return picked.sort((a, b) => a.start - b.start).map((candidate, index) => ({
    ...candidate,
    rank: index + 1,
  }));
}

export function assTime(seconds) {
  const totalCs = Math.max(0, Math.round(seconds * 100));
  const hours = Math.floor(totalCs / 360000);
  const minutes = Math.floor((totalCs % 360000) / 6000);
  const secs = Math.floor((totalCs % 6000) / 100);
  const centis = totalCs % 100;
  return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(centis).padStart(2, "0")}`;
}

export function assEscape(text) {
  return String(text || "")
    .replace(/[\r\n]+/g, " ")
    .replace(/[{}]/g, (match) => (match === "{" ? "(" : ")"))
    .replace(/\\/g, "/")
    .replace(/\s+/g, " ")
    .trim();
}

function captionGroups(words, clipStart, duration) {
  const maxWords = 4;
  const maxChars = 22;
  const maxSeconds = 3;
  const clipped = words
    .filter((word) => word.end > clipStart + 0.01 && word.start < clipStart + duration - 0.01)
    .map((word) => ({
      start: Math.max(0, word.start - clipStart),
      end: Math.min(duration, word.end - clipStart),
      text: word.text,
    }))
    .sort((a, b) => a.start - b.start);

  const groups = [];
  let current = [];
  let chars = 0;
  for (const word of clipped) {
    if (current.length) {
      const last = current[current.length - 1];
      const gap = word.start - last.end;
      const terminal = TERMINAL_RE.test(last.text);
      const tooLong = last.end - current[0].start >= maxSeconds;
      const tooMany = current.length >= maxWords;
      const tooWide = chars + 1 + word.text.length > maxChars;
      if (gap > 0.55 || terminal || tooLong || tooMany || tooWide) {
        groups.push(current);
        current = [];
        chars = 0;
      }
    }
    current.push(word);
    chars += word.text.length + (chars ? 1 : 0);
    if (TERMINAL_RE.test(word.text)) {
      groups.push(current);
      current = [];
      chars = 0;
    }
  }
  if (current.length) groups.push(current);

  for (let i = 0; i < groups.length - 1; i += 1) {
    const last = groups[i][groups[i].length - 1];
    const nextStart = groups[i + 1][0].start - 0.04;
    if (last.end + 0.22 > nextStart) {
      last.end = Math.max(last.start + 0.04, nextStart - 0.26);
    }
  }
  return groups;
}

export function buildAss(words, clipStart, duration) {
  const groups = captionGroups(words, clipStart, duration);
  const events = [];
  let previousEnd = 0;

  for (const group of groups) {
    const cueStart = Math.max(0, group[0].start - 0.04, previousEnd);
    const cueEnd = Math.min(duration, group[group.length - 1].end + 0.22);
    if (cueEnd <= cueStart) continue;
    const boundaries = [cueStart];
    for (const word of group.slice(1)) {
      boundaries.push(Math.max(cueStart, Math.min(cueEnd, word.start)));
    }
    boundaries.push(cueEnd);

    for (let active = 0; active < group.length; active += 1) {
      const eventStart = boundaries[active];
      let eventEnd = boundaries[active + 1];
      if (eventEnd - eventStart < 0.03) eventEnd = Math.min(cueEnd, eventStart + 0.05);
      if (eventEnd <= eventStart) continue;
      const line = group
        .map((word, index) => {
          const token = assEscape(word.text.toUpperCase());
          if (!token) return "";
          if (index === active) return `{\\c&H00FFFF&}${token}{\\c&H00FFFFFF&}`;
          return token;
        })
        .filter(Boolean)
        .join(" ");
      if (line) {
        events.push(
          `Dialogue: 0,${assTime(eventStart)},${assTime(eventEnd)},Viral,,0,0,0,,${line}`
        );
      }
    }
    previousEnd = cueEnd;
  }

  return [
    "[Script Info]",
    "; Generated by Viral Clip Forge browser demo",
    "ScriptType: v4.00+",
    "PlayResX: 1080",
    "PlayResY: 1920",
    "WrapStyle: 2",
    "ScaledBorderAndShadow: yes",
    "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    "Style: Viral,Noto Sans,66,&H00FFFFFF,&H00FFFF00,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,60,60,220,1",
    "",
    "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ...events,
    "",
  ].join("\n");
}
