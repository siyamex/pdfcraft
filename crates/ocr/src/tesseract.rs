//! Recognition through an installed Tesseract OCR program, for scripts the ocrs models don't read
//! (Dhivehi, written in Thaana). Fork-only: upstream PdfCraft is Rust only and ships no external
//! engine.
//!
//! Tesseract's own page layout analysis loses Thaana: its vowel signs (fili) float above and below
//! the letters as separate strokes, and a page of clean Faruma came back as noise or nothing. So
//! the lines are found here ([`text_lines`]: rows of ink, with the thin rows of fili joined to
//! the letters beside them) and Tesseract reads each one as a single line (`--psm 7`), all in one
//! run through a list of line images. Its TSV output gives each word's text (in logical order)
//! and box. Tesseract runs without a console window and is stopped after [`TIMEOUT`]. Its language data (`div.traineddata` for Dhivehi) is looked
//! for in `$PDFCRAFT_TESSDATA`, `$TESSDATA_PREFIX`, a `tessdata` folder beside the ocrs models
//! ([`crate::Models::search_dirs`]), then wherever Tesseract keeps its own.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{Line, OcrError, Word};

/// A page that takes longer than this is abandoned (Tesseract reads an A4 page in seconds).
const TIMEOUT: Duration = Duration::from_secs(180);
/// Larger TSV output than this is not read: the process output is untrusted input too.
const MAX_OUTPUT: u64 = 64 << 20;

/// A Tesseract program and the language it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Tesseract {
    pub exe: PathBuf,
    /// Tesseract's language code (`div` for Dhivehi).
    pub language: String,
    /// A folder holding `<language>.traineddata`, when it isn't Tesseract's own.
    pub tessdata: Option<PathBuf>,
}

impl Tesseract {
    /// Find Tesseract and its data for `language`.
    pub fn find(language: &str) -> Result<Tesseract, OcrError> {
        let exe = find_exe().ok_or(OcrError::NoTesseract)?;
        let file = format!("{language}.traineddata");
        let tessdata = tessdata_dirs().into_iter().find(|d| d.join(&file).is_file());
        if tessdata.is_none() && !exe.parent().is_some_and(|d| d.join("tessdata").join(&file).is_file()) {
            return Err(OcrError::NoTesseractLanguage(language.to_owned()));
        }
        Ok(Tesseract { exe, language: language.to_owned(), tessdata })
    }

    /// Recognise the text of an RGB(A) image.
    pub fn recognize(&self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        let gray = gray(pixels, width, height).ok_or(OcrError::EmptyImage)?;
        let lines = text_lines(&gray, width, height);
        let mut read: Vec<Option<Line>> = vec![None; lines.len()];
        // Each line at the height Tesseract reads best; a line it returns nothing for (the
        // Dhivehi model drops the odd line outright, and a pixel's difference brings it back) is
        // read again at the next height.
        let mut pending: Vec<usize> = (0..lines.len()).collect();
        for line_height in LINE_HEIGHTS {
            if pending.is_empty() {
                break;
            }
            let images: Vec<(usize, LineImage)> =
                pending.iter().filter_map(|&i| Some((i, line_image(&gray, width, *lines.get(i)?, line_height)?))).collect();
            if images.is_empty() {
                break;
            }
            for (page, mut line) in self.read_lines(&images.iter().map(|(_, im)| im).collect::<Vec<_>>())? {
                let Some((i, image)) = page.checked_sub(1).and_then(|p| images.get(p as usize)) else { continue };
                let Some(&(top, bottom)) = lines.get(*i) else { continue };
                let (border, [x0, _], scale) = (BORDER as f32, image.origin, image.scale);
                for w in &mut line.words {
                    let [l, _, r, _] = w.rect;
                    let x = |v: f32| (x0 + (v - border) * scale).clamp(0.0, width as f32);
                    // Every word spans its whole line: a word's own box ends higher or lower with
                    // its fili, and words on different baselines would read as different lines.
                    w.rect = [x(l), top as f32, x(r), bottom as f32];
                    w.text = clean(&w.text);
                }
                line.words.retain(|w| !w.text.is_empty());
                if !line.words.is_empty()
                    && let Some(slot) = read.get_mut(*i)
                {
                    *slot = Some(line);
                }
            }
            pending.retain(|&i| read.get(i).is_some_and(Option::is_none));
        }
        Ok(read.into_iter().flatten().collect())
    }

    /// Tesseract's lines for `images`, one line image per "page", with the page number (from 1).
    fn read_lines(&self, images: &[&LineImage]) -> Result<Vec<(u32, Line)>, OcrError> {
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let mut files = Vec::with_capacity(images.len());
        for image in images {
            let file = TempFile::new("pgm")?;
            std::fs::write(&file.0, &image.pgm).map_err(|e| err(&e))?;
            files.push(file);
        }
        let list = TempFile::new("txt")?;
        let listing: String = files.iter().map(|f| format!("{}\n", f.0.display())).collect();
        std::fs::write(&list.0, listing).map_err(|e| err(&e))?;
        let mut cmd = Command::new(&self.exe);
        cmd.arg(&list.0).arg("stdout").arg("-l").arg(&self.language);
        if let Some(dir) = &self.tessdata {
            cmd.arg("--tessdata-dir").arg(dir);
        }
        // Each image is one line; TSV output gives word boxes, one "page" per line image. Set as a
        // variable, not the `tsv` config file, which `--tessdata-dir` would look for in that folder.
        cmd.args(["--psm", "7", "-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=0"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            // CREATE_NO_WINDOW: no console flashes up for each page.
            cmd.creation_flags(0x0800_0000);
        }
        let mut child = cmd.spawn().map_err(|e| OcrError::Recognize(format!("starting {}: {e}", self.exe.display())))?;
        let mut stdout = child.stdout.take().ok_or_else(|| OcrError::Recognize("no output from tesseract".into()))?;
        // Read on a thread, so a full pipe can't stall the wait below.
        let reader = std::thread::spawn(move || {
            let mut out = Vec::new();
            (&mut stdout).take(MAX_OUTPUT).read_to_end(&mut out).map(|_| out)
        });
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() > TIMEOUT => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(OcrError::Recognize(format!("tesseract took longer than {} seconds", TIMEOUT.as_secs())));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => return Err(err(&e)),
            }
        };
        let out = reader.join().map_err(|_| OcrError::Recognize("reading tesseract's output failed".into()))?.map_err(|e| err(&e))?;
        if !status.success() {
            let mut message = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = (&mut e).take(4096).read_to_string(&mut message);
            }
            return Err(OcrError::Recognize(format!("tesseract failed ({status}): {}", message.trim())));
        }
        Ok(parse_tsv(&String::from_utf8_lossy(&out)))
    }
}

/// A recognised word without what Tesseract's Dhivehi model scatters into its output but Thaana
/// never uses: Arabic vowel marks (U+064B–U+065F, a stray kasra after many words) and the
/// invisible direction marks and embeddings (U+200E, U+200F, U+202A–U+202E).
fn clean(word: &str) -> String {
    word.chars()
        .filter(|c| !matches!(c, '\u{064B}'..='\u{065F}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}'))
        .collect::<String>()
        .trim()
        .to_owned()
}

/// `pixels` (RGBA, RGB or grey, `width` × `height`) as one grey byte per pixel. `None` for a
/// size that doesn't fit.
fn gray(pixels: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let count = usize::try_from(u64::from(width).checked_mul(u64::from(height))?).ok()?;
    if count == 0 {
        return None;
    }
    let channels = pixels.len().checked_div(count)?;
    if !matches!(channels, 1 | 3 | 4) || pixels.len() != count.checked_mul(channels)? {
        return None;
    }
    Some(
        pixels
            .chunks_exact(channels)
            .map(|p| match p {
                [r, g, b, ..] => ((u32::from(*r) * 299 + u32::from(*g) * 587 + u32::from(*b) * 114) / 1000) as u8,
                [v, ..] => *v,
                [] => 255,
            })
            .collect(),
    )
}

/// The most lines one page is read as.
const MAX_LINES: usize = 1000;

/// The text lines of a grey image, top to bottom, as `(top, bottom)` pixel rows (bottom
/// exclusive), from the bands of rows with ink (darker than mid-grey). A Thaana line is three
/// bands, its fili above, its letters, its fili below, and the fili bands can be nearly as tall
/// as the letters (stacked marks), so heights don't tell them apart. The gaps do: a few pixels
/// inside a line, many between lines (and more between paragraphs). The gaps are split in two by
/// Otsu's method on their logarithms, if the two gaps either side of the split differ at least
/// 2.5 times; without such a split (a single line, no clear spacing), [`join_by_height`] decides.
fn text_lines(gray: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    let bands = ink_bands(gray, width, height);
    let gaps: Vec<u32> = bands.windows(2).filter_map(|p| Some(p.get(1)?.0.saturating_sub(p.first()?.1).max(1))).collect();
    let cut = gap_split(&gaps);
    let mut lines = match cut {
        Some(cut) => {
            let mut lines: Vec<(u32, u32)> = Vec::new();
            for (i, &band) in bands.iter().enumerate() {
                match (lines.last_mut(), i.checked_sub(1).and_then(|g| gaps.get(g))) {
                    (Some(line), Some(&gap)) if gap < cut => line.1 = band.1,
                    _ => lines.push(band),
                }
            }
            lines
        }
        None => join_by_height(&bands),
    };
    lines.truncate(MAX_LINES);
    lines
}

/// The gap size below which bands are one line: Otsu's split of the gaps' logarithms (the split
/// with the largest between-group variance), when the gaps either side of it differ at least 2.5
/// times. `None` without such a split.
fn gap_split(gaps: &[u32]) -> Option<u32> {
    let mut sorted: Vec<u32> = gaps.to_vec();
    sorted.sort_unstable();
    let logs: Vec<f64> = sorted.iter().map(|&g| f64::from(g.max(1)).ln()).collect();
    let n = logs.len() as f64;
    let total: f64 = logs.iter().sum();
    let mut best: Option<(f64, usize)> = None;
    let mut below = 0.0;
    for k in 1..logs.len() {
        below += logs.get(k - 1).copied().unwrap_or(0.0);
        let (n0, n1) = (k as f64, n - k as f64);
        let variance = n0 * n1 * (below / n0 - (total - below) / n1).powi(2);
        if best.is_none_or(|(v, _)| variance > v) {
            best = Some((variance, k));
        }
    }
    let (_, k) = best?;
    let (small, large) = (*sorted.get(k - 1)?, *sorted.get(k)?);
    (u64::from(large) * 10 >= u64::from(small) * 25).then(|| small.saturating_add(large).div_ceil(2))
}

/// The bands of rows with ink, top to bottom.
fn ink_bands(gray: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    let w = width as usize;
    // A row needs a few dark pixels to count, so specks of scan noise don't make lines.
    let min_ink = (w / 1000).max(2);
    let inked: Vec<bool> =
        (0..height as usize).map(|y| gray.get(y * w..(y + 1) * w).is_some_and(|row| row.iter().filter(|&&v| v < 128).count() >= min_ink)).collect();
    let mut bands: Vec<(u32, u32)> = Vec::new();
    let mut y = 0usize;
    while y < inked.len() {
        if inked.get(y).copied().unwrap_or(false) {
            let start = y;
            while inked.get(y).copied().unwrap_or(false) {
                y += 1;
            }
            bands.push((start as u32, y as u32));
        } else {
            y += 1;
        }
    }
    bands
}

/// Lines from `bands` without clear spacing: a band joins the line of an adjacent band at least
/// 1.3 times as tall and no further away than half that band's height (fili beside their
/// letters); lines of letters, about as tall as each other, stay apart.
fn join_by_height(bands: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let tall = |b: (u32, u32)| u64::from(b.1 - b.0);
    // Each band joins the closer adjacent band that is clearly taller and near enough.
    let mut owner: Vec<usize> = (0..bands.len()).collect();
    for (i, &band) in bands.iter().enumerate() {
        let fits = |j: usize| {
            let other = *bands.get(j)?;
            let gap = u64::from(if j < i { band.0.saturating_sub(other.1) } else { other.0.saturating_sub(band.1) });
            (tall(other) * 10 >= tall(band) * 13 && gap * 2 <= tall(other)).then_some((gap, j))
        };
        if let Some((_, j)) = [i.checked_sub(1).and_then(fits), fits(i + 1)].into_iter().flatten().min() {
            owner[i] = j;
        }
    }
    let mut lines: Vec<(usize, (u32, u32))> = Vec::new();
    for (i, &band) in bands.iter().enumerate() {
        // Follow the chain to the band that joined none (each step goes to a taller band, so it ends).
        let mut root = i;
        for _ in 0..bands.len() {
            match owner.get(root) {
                Some(&next) if next != root => root = next,
                _ => break,
            }
        }
        match lines.iter_mut().find(|(r, _)| *r == root) {
            Some((_, l)) => *l = (l.0.min(band.0), l.1.max(band.1)),
            None => lines.push((root, band)),
        }
    }
    let mut lines: Vec<(u32, u32)> = lines.into_iter().map(|(_, l)| l).collect();
    lines.sort_unstable();
    lines
}

/// The heights, in pixels, each line is scaled to before Tesseract reads it, the first first and
/// the others for lines it returned nothing for. Measured on 22 lines of Faruma scanned at 300
/// dpi: 27% of characters wrong as scanned, 16% at 48 px, 17% at 44 px, 19% at 40 px, 22% at
/// 56 px.
const LINE_HEIGHTS: [u32; 3] = [48, 40, 56];
/// White border around each line image, in pixels.
const BORDER: u32 = 16;

/// A line cut out of the page for Tesseract: its image, and how to map back to the page.
struct LineImage {
    pgm: Vec<u8>,
    /// The page pixel of the line image's ink origin (after the border).
    origin: [f32; 2],
    /// Page pixels per line-image pixel.
    scale: f32,
}

/// The ink of `rows` (top, bottom) of the page, cropped to its extent, scaled to `line_height`
/// pixels and framed in a white [`BORDER`], as a binary PGM. `None` without ink.
fn line_image(gray: &[u8], width: u32, (top, bottom): (u32, u32), line_height: u32) -> Option<LineImage> {
    let w = width as usize;
    let dark = |x: usize, y: usize| gray.get(y * w + x).is_some_and(|&v| v < 128);
    let rows = top as usize..bottom as usize;
    let left = (0..w).find(|&x| rows.clone().any(|y| dark(x, y)))?;
    let right = (0..w).rev().find(|&x| rows.clone().any(|y| dark(x, y)))? + 1;
    let (cw, ch) = (right - left, rows.len());
    if ch == 0 {
        return None;
    }
    let line_height = line_height.max(1);
    let scale = ch as f32 / line_height as f32;
    let nw = ((cw as f32 / scale).round() as usize).clamp(1, 16_384);
    let nh = line_height as usize;
    let at = |x: usize, y: usize| f32::from(gray.get((top as usize + y) * w + left + x).copied().unwrap_or(255));
    let (bw, bh) = (nw + 2 * BORDER as usize, nh + 2 * BORDER as usize);
    let mut out = vec![255u8; bw * bh];
    for ny in 0..nh {
        for nx in 0..nw {
            // Shrinking averages the source pixels under the target pixel; enlarging interpolates.
            let (x0, x1) = (nx as f32 * scale, (nx + 1) as f32 * scale);
            let (y0, y1) = (ny as f32 * scale, (ny + 1) as f32 * scale);
            let v = if scale >= 1.0 {
                let (xs, xe) = (x0 as usize, (x1.ceil() as usize).clamp(x0 as usize + 1, cw));
                let (ys, ye) = (y0 as usize, (y1.ceil() as usize).clamp(y0 as usize + 1, ch));
                let mut sum = 0.0;
                for y in ys..ye {
                    for x in xs..xe {
                        sum += at(x, y);
                    }
                }
                sum / ((xe - xs) * (ye - ys)).max(1) as f32
            } else {
                let (fx, fy) = (((x0 + x1) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32), ((y0 + y1) / 2.0 - 0.5).clamp(0.0, (ch - 1) as f32));
                let (ix, iy) = (fx as usize, fy as usize);
                let (tx, ty) = (fx - ix as f32, fy - iy as f32);
                let (ix1, iy1) = ((ix + 1).min(cw - 1), (iy + 1).min(ch - 1));
                (at(ix, iy) * (1.0 - tx) + at(ix1, iy) * tx) * (1.0 - ty) + (at(ix, iy1) * (1.0 - tx) + at(ix1, iy1) * tx) * ty
            };
            if let Some(p) = out.get_mut((ny + BORDER as usize) * bw + nx + BORDER as usize) {
                *p = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut pgm = format!("P5\n{bw} {bh}\n255\n").into_bytes();
    pgm.extend_from_slice(&out);
    Some(LineImage { pgm, origin: [left as f32, top as f32], scale })
}

/// Words from Tesseract's TSV output (`level page block par line word left top width height conf
/// text`), grouped into lines in the order Tesseract reads them, each with its page number.
pub(crate) fn parse_tsv(tsv: &str) -> Vec<(u32, Line)> {
    let mut lines: Vec<((u32, u32, u32, u32), Line)> = Vec::new();
    for row in tsv.lines().skip(1) {
        let f: Vec<&str> = row.splitn(12, '\t').collect();
        let [level, page, block, par, line, _, left, top, w, h, _conf, text] = f.as_slice() else { continue };
        let num = |s: &str| s.trim().parse::<u32>().ok();
        let (Some(5), Some(page), Some(block), Some(par), Some(line)) = (num(level), num(page), num(block), num(par), num(line)) else { continue };
        let (Some(l), Some(t), Some(w), Some(h)) = (num(left), num(top), num(w), num(h)) else { continue };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let word = Word { text: text.to_owned(), rect: [l as f32, t as f32, l.saturating_add(w) as f32, t.saturating_add(h) as f32] };
        let key = (page, block, par, line);
        match lines.last_mut() {
            Some((k, l)) if *k == key => l.words.push(word),
            _ => lines.push((key, Line { words: vec![word] })),
        }
    }
    lines.into_iter().map(|((page, ..), l)| (page, l)).collect()
}

/// The Tesseract program: `$PDFCRAFT_TESSERACT`, `tesseract` on the `PATH`, then where the
/// Windows installers put it.
fn find_exe() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PDFCRAFT_TESSERACT").map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let name = if cfg!(windows) { "tesseract.exe" } else { "tesseract" };
    let on_path = std::env::var_os("PATH").into_iter().flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).map(|d| d.join(name));
    let installed = if cfg!(windows) {
        ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"]
            .iter()
            .filter_map(std::env::var_os)
            .flat_map(|d| [Path::new(&d).join("Tesseract-OCR").join(name), Path::new(&d).join("Programs").join("Tesseract-OCR").join(name)])
            .collect()
    } else {
        vec![PathBuf::from("/opt/homebrew/bin/tesseract"), PathBuf::from("/usr/local/bin/tesseract"), PathBuf::from("/usr/bin/tesseract")]
    };
    on_path.chain(installed).find(|p| p.is_file())
}

/// Folders that may hold language data, best first.
fn tessdata_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = ["PDFCRAFT_TESSDATA", "TESSDATA_PREFIX"].iter().filter_map(std::env::var_os).map(PathBuf::from).collect();
    dirs.extend(crate::Models::search_dirs().into_iter().map(|d| d.join("tessdata")));
    dirs
}

/// A file in the temporary folder, removed when dropped.
struct TempFile(PathBuf);

impl TempFile {
    fn new(extension: &str) -> Result<TempFile, OcrError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Ok(TempFile(std::env::temp_dir().join(format!("pdfcraft-ocr-{}-{n}.{extension}", std::process::id()))))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsv_words_are_grouped_into_lines() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   1\t1\t0\t0\t0\t0\t0\t0\t800\t600\t-1\t\n\
                   4\t1\t1\t1\t1\t0\t10\t10\t300\t40\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t200\t10\t110\t40\t91.5\tދިވެހި\n\
                   5\t1\t1\t1\t1\t2\t10\t12\t150\t38\t88.0\tބަސް\n\
                   5\t1\t1\t1\t2\t1\t10\t70\t90\t40\t70.0\t  \n\
                   5\t1\t1\t1\t2\t2\t10\t70\t90\t40\t70.0\tރާއްޖެ\n\
                   5\tx\t1\t1\t2\t3\t10\t70\t90\t40\t70.0\tbad\n\
                   5\t1\t1\n";
        let lines = parse_tsv(tsv);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].1.text(), "ދިވެހި ބަސް");
        assert_eq!(lines[0].1.words[0].rect, [200.0, 10.0, 310.0, 50.0]);
        assert_eq!(lines[1].1.text(), "ރާއްޖެ");
        assert_eq!((lines[0].0, lines[1].0), (1, 1));
        assert!(parse_tsv("").is_empty());
        // A box past u32 saturates instead of overflowing.
        let huge = format!("h\n5\t1\t1\t1\t1\t1\t{0}\t{0}\t{0}\t{0}\t1\tx\n", u32::MAX);
        assert_eq!(parse_tsv(&huge)[0].1.words[0].rect[2], u32::MAX as f32);
    }

    #[test]
    fn images_become_grey() {
        assert_eq!(gray(&[255, 255, 255, 255, 0, 0, 0, 255], 2, 1), Some(vec![255, 0]));
        assert_eq!(gray(&[10, 200], 2, 1), Some(vec![10, 200]));
        assert_eq!(gray(&[1, 2, 3], 2, 1), None, "too few pixels");
        assert_eq!(gray(&[], 0, 0), None);
        assert_eq!(gray(&[0; 8], u32::MAX, u32::MAX), None);
    }

    /// A page drawn as bands of ink rows: `(top, height)` each, 100 pixels wide.
    fn page(bands: &[(u32, u32)], height: u32) -> Vec<u8> {
        let mut px = vec![255u8; 100 * height as usize];
        for &(top, h) in bands {
            for y in top..top + h {
                px[y as usize * 100..y as usize * 100 + 50].fill(0);
            }
        }
        px
    }

    #[test]
    fn fili_above_and_below_join_their_letters_line() {
        // Two Thaana lines: fili above (4 px), a 2 px gap, letters (20 px), a 2 px gap, fili below.
        let bands = [(10, 4), (16, 20), (38, 4), (60, 4), (66, 20), (88, 4)];
        assert_eq!(text_lines(&page(&bands, 120), 100, 120), [(10, 42), (60, 92)]);
    }

    /// The ink bands of a 300 dpi scan of three lines of Faruma (the first test page): fili above,
    /// letters, fili below, the fili up to two thirds of the letters' height.
    #[test]
    fn a_scanned_pages_fili_join_their_lines() {
        let bands = [(322, 25), (356, 40), (399, 17), (548, 21), (580, 38), (619, 24), (766, 25), (800, 38), (840, 25)];
        assert_eq!(text_lines(&page(&bands, 900), 100, 900), [(322, 416), (548, 643), (766, 865)]);
    }

    /// Stacked fili can make a fili band nearly as tall as the letters (27 and 32 px on a scan of
    /// Faruma): the gaps, a few pixels inside a line and about fifty between lines, still group them.
    #[test]
    fn lines_are_grouped_by_their_gaps() {
        let bands = [(306, 19), (333, 31), (365, 19), (432, 63), (498, 13), (957, 27), (989, 32), (1023, 13), (1093, 59)];
        let lines = text_lines(&page(&bands, 1200), 100, 1200);
        assert_eq!(lines, [(306, 384), (432, 511), (957, 1036), (1093, 1152)]);
    }

    #[test]
    fn stray_arabic_marks_and_direction_marks_are_removed() {
        assert_eq!(clean("ދީންِ"), "ދީން");
        assert_eq!(clean("\u{200F}ސަރުކާރު\u{202C}"), "ސަރުކާރު");
        assert_eq!(clean("ِ"), "");
        // Thaana fili, digits and punctuation stay.
        assert_eq!(clean("ދިވެހި 2018:"), "ދިވެހި 2018:");
    }

    #[test]
    fn gaps_split_between_lines_not_at_a_paragraph_break() {
        // In-line gaps 1–8, line gaps about 50, one paragraph break.
        assert_eq!(gap_split(&[8, 1, 48, 3, 446, 5, 2, 57]), Some(28));
        // No clear second group: no split.
        assert_eq!(gap_split(&[10, 12, 11, 9]), None);
        assert_eq!(gap_split(&[7]), None);
        assert_eq!(gap_split(&[]), None);
        assert_eq!(gap_split(&[u32::MAX, 1]), Some(u32::MAX / 2 + 1));
    }

    #[test]
    fn a_line_is_cropped_to_its_ink_and_scaled_to_the_reading_height() {
        // A 200 × 96 px dark block at x 300..500, y 100..196, on a 1000 px wide page.
        let (w, h) = (1000usize, 300usize);
        let mut gray = vec![255u8; w * h];
        for y in 100..196 {
            gray[y * w + 300..y * w + 500].fill(0);
        }
        let image = line_image(&gray, w as u32, (100, 196), 48).unwrap();
        // 96 px tall becomes 48 (scale 2), 200 px wide becomes 100, plus the border.
        assert_eq!(image.scale, 2.0);
        assert_eq!(image.origin, [300.0, 100.0]);
        let header = format!("P5\n{} {}\n255\n", 100 + 2 * BORDER, 48 + 2 * BORDER);
        assert!(image.pgm.starts_with(header.as_bytes()));
        let px = &image.pgm[header.len()..];
        let row = (BORDER as usize + 24) * (100 + 2 * BORDER as usize);
        assert_eq!(px[row + BORDER as usize + 50], 0, "ink inside");
        assert_eq!(px[row + 2], 255, "white border");
        // A small line is enlarged, never empty.
        let mut small = vec![255u8; 100 * 20];
        small[5 * 100 + 10..5 * 100 + 30].fill(0);
        let image = line_image(&small, 100, (5, 6), 48).unwrap();
        assert!(image.scale < 1.0 && image.pgm.len() > 100);
        // Each retry height gives its own size; a zero height is one pixel, never a division by zero.
        assert!(
            line_image(&gray, w as u32, (100, 196), 40).unwrap().pgm.starts_with(format!("P5\n{} {}\n", 83 + 2 * BORDER, 40 + 2 * BORDER).as_bytes())
        );
        assert!(line_image(&gray, w as u32, (100, 196), 0).is_some());
        // No ink, no image.
        assert!(line_image(&vec![255u8; 100 * 20], 100, (0, 20), 48).is_none());
        assert!(line_image(&[], 100, (0, 20), 48).is_none());
    }

    #[test]
    fn close_lines_of_letters_stay_apart() {
        // Lines of letters 2 px apart are two lines, however close.
        assert_eq!(text_lines(&page(&[(10, 20), (32, 20)], 60), 100, 60), [(10, 30), (32, 52)]);
        // A thin band far from any letters is its own line (a rule, a dash).
        assert_eq!(text_lines(&page(&[(10, 20), (80, 3)], 100), 100, 100), [(10, 30), (80, 83)]);
        // Blank and broken pages have no lines.
        assert!(text_lines(&page(&[], 50), 100, 50).is_empty());
        assert!(text_lines(&[], 100, 50).is_empty());
        // A speck of noise (one dark pixel per row) is not text.
        let mut speck = vec![255u8; 100 * 10];
        speck[500] = 0;
        assert!(text_lines(&speck, 100, 10).is_empty());
    }

    #[test]
    fn temporary_files_are_removed() {
        let path = {
            let f = TempFile::new("ppm").unwrap();
            std::fs::write(&f.0, b"x").unwrap();
            f.0.clone()
        };
        assert!(!path.exists());
    }
}
