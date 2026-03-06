use std::{
    collections::BTreeMap,
    f64::consts::FRAC_1_SQRT_2,
    path::Path,
};

use divan_parser::BenchResult;
use plotters::{
    element::{
        Drawable,
        PointCollection,
    },
    prelude::{
        IntoLogRange,
        ShapeStyle,
        *,
    },
    style::{
        SizeDesc,
        full_palette as colors,
    },
};
use plotters_backend::{
    BackendCoord,
    DrawingErrorKind,
};
use wordchipper_cli_util::logging::LogArgs;

pub const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;

use crate::util::bench_data::par_bench::ParBenchData;

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct DevArgs {
    /// Path to the benchmark data.
    #[clap(long, default_value = "dev-crates/wordchipper-bench/bench-data")]
    data_dir: String,

    /// Model name.
    #[clap(long, default_value = "cl100k")]
    model: String,

    /// Path to the output directory.
    #[clap(long, default_value = "target/plots")]
    output_dir: String,

    #[clap(flatten)]
    logging: LogArgs,
}

impl DevArgs {
    /// Run the dev command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        println!("{:?}", self);

        let data_dir = Path::new(&self.data_dir);

        let data = ParBenchData::load_data(data_dir.join("parallel"))?;

        let output_dir = Path::new(&self.output_dir);
        std::fs::create_dir_all(output_dir)?;

        build_demo_graph(&output_dir)?;

        build_external_tgraph(&self.model, "buffer_sweep", &output_dir, &data)?;

        for accel in [false, true] {
            build_internal_tgraph(
                &self.model,
                accel,
                &output_dir.join(format!(
                    "tgraph.{}.{}.svg",
                    if accel { "logos" } else { "regex" },
                    self.model
                )),
                &data,
            )?;
        }
        for accel in [false, true] {
            build_internal_rel_tgraph(
                &self.model,
                accel,
                &output_dir.join(format!(
                    "tgraph.rel.{}.{}.svg",
                    if accel { "logos" } else { "regex" },
                    self.model
                )),
                &data,
            )?;
        }

        Ok(())
    }
}

struct Point {
    pub threads: u32,
    pub value: f64,
}

struct Series {
    pub name: String,
    pub glyph: String,
    pub style: ShapeStyle,
    pub points: Vec<Point>,
}
impl Series {
    pub fn min_threads(&self) -> u32 {
        self.points.iter().map(|p| p.threads).min().unwrap()
    }

    pub fn max_threads(&self) -> u32 {
        self.points.iter().map(|p| p.threads).max().unwrap()
    }

    pub fn min_bps(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    pub fn max_bps(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

fn external_styles() -> BTreeMap<String, (String, ShapeStyle)> {
    [
        (
            "bpe_openai".to_string(),
            ("♣︎".to_string(), colors::DEEPORANGE_200.filled()),
        ),
        (
            "tiktoken".to_string(),
            ("♥︎".to_string(), colors::PURPLE_200.filled()),
        ),
        (
            "tokenizers".to_string(),
            ("♦︎".to_string(), colors::PINK_200.filled()),
        ),
    ]
    .iter()
    .cloned()
    .collect()
}

fn fmin(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => a,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => b,
    }
}
fn fmax(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => a,
    }
}

fn median_bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().median.unwrap()
}

fn as_points<F>(
    obs: &[(u32, BenchResult)],
    f: F,
) -> Vec<Point>
where
    F: Fn(u32, &BenchResult) -> f64,
{
    obs.iter()
        .map(|(threads, br)| Point {
            threads: *threads,
            value: f(*threads, br),
        })
        .collect()
}

pub struct AbstractPath {
    pub path: Vec<(f64, f64)>,
}

impl<I> From<I> for AbstractPath
where
    I: IntoIterator<Item = (f64, f64)>,
{
    fn from(value: I) -> Self {
        Self {
            path: value.into_iter().collect(),
        }
    }
}

impl AbstractPath {
    pub fn to_size(
        &self,
        size: u32,
    ) -> Vec<BackendCoord> {
        self.path
            .iter()
            .map(|(x, y)| {
                let x = (x * size as f64) as i32;
                let y = (y * size as f64) as i32;
                (x, y)
            })
            .collect()
    }
}

pub struct PathMarker<Coord, Size: SizeDesc> {
    coord: Coord,
    size: Size,
    path: AbstractPath,
    stroke: Option<ShapeStyle>,
    fill: Option<ShapeStyle>,
}

impl<Coord, Size: SizeDesc> PathMarker<Coord, Size> {
    pub fn new<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
        path: AbstractPath,
    ) -> Self {
        Self {
            coord,
            size,
            path,
            stroke: stroke.into(),
            fill: fill.into(),
        }
    }

    pub fn square<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const STEP: f64 = FRAC_1_SQRT_2;
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![(-STEP, -STEP), (STEP, -STEP), (STEP, STEP), (-STEP, STEP)].into(),
        )
    }

    pub fn cross_square<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const ORIGIN: (f64, f64) = (0.0, 0.0);
        const STEP: f64 = FRAC_1_SQRT_2;
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (-STEP, -STEP),
                (0.0, -STEP),
                ORIGIN,
                (0.0, -STEP),
                (STEP, -STEP),
                (STEP, 0.0),
                ORIGIN,
                (STEP, 0.0),
                (STEP, STEP),
                (0.0, STEP),
                ORIGIN,
                (0.0, STEP),
                (-STEP, STEP),
                (-STEP, 0.0),
                ORIGIN,
                (-STEP, 0.0),
            ]
            .into(),
        )
    }

    pub fn diamond<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const STEP: f64 = 1.0;
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![(0.0, -STEP), (STEP, 0.0), (0.0, STEP), (-STEP, 0.0)].into(),
        )
    }

    pub fn cross_diamond<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const STEP: f64 = 0.5;
        const ORIGIN: (f64, f64) = (0.0, 0.0);
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, -1.0),
                (STEP, -STEP),
                ORIGIN,
                (STEP, -STEP),
                (1.0, 0.0),
                (STEP, STEP),
                ORIGIN,
                (STEP, STEP),
                (0.0, 1.0),
                (-STEP, STEP),
                ORIGIN,
                (-STEP, STEP),
                (-1.0, 0.0),
                (-STEP, -STEP),
                ORIGIN,
                (-STEP, -STEP),
            ]
            .into(),
        )
    }

    pub fn star<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const STEP: f64 = FRAC_1_SQRT_2 / 2.0;
        const ORIGIN: (f64, f64) = (0.0, 0.0);
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, -1.0),
                (STEP, -STEP),
                ORIGIN,
                (STEP, -STEP),
                (1.0, 0.0),
                (STEP, STEP),
                ORIGIN,
                (STEP, STEP),
                (0.0, 1.0),
                (-STEP, STEP),
                ORIGIN,
                (-STEP, STEP),
                (-1.0, 0.0),
                (-STEP, -STEP),
                ORIGIN,
                (-STEP, -STEP),
            ]
            .into(),
        )
    }

    pub fn tri_up<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, -1.0),
                (1.5 / SQRT_3, 1.0 / 2.0),
                (-1.5 / SQRT_3, 1.0 / 2.0),
            ]
            .into(),
        )
    }

    pub fn cross_tri_up<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const ORIGIN: (f64, f64) = (0.0, 0.0);

        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, -1.0),
                (1.5 / SQRT_3 / 2.0, -0.25),
                ORIGIN,
                (1.5 / SQRT_3 / 2.0, -0.25),
                (1.5 / SQRT_3, 1.0 / 2.0),
                (0.0, 1.0 / 2.0),
                ORIGIN,
                (0.0, 1.0 / 2.0),
                (-1.5 / SQRT_3, 1.0 / 2.0),
                (-1.5 / SQRT_3 / 2.0, -0.25),
                ORIGIN,
                (-1.5 / SQRT_3 / 2.0, -0.25),
            ]
            .into(),
        )
    }

    pub fn tri_down<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, 1.0),
                (1.5 / SQRT_3, -1.0 / 2.0),
                (-1.5 / SQRT_3, -1.0 / 2.0),
            ]
            .into(),
        )
    }

    pub fn cross_tri_down<S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
    ) -> Self {
        const ORIGIN: (f64, f64) = (0.0, 0.0);

        Self::new(
            coord,
            size,
            stroke,
            fill,
            vec![
                (0.0, 1.0),
                (1.5 / SQRT_3 / 2.0, 0.25),
                ORIGIN,
                (1.5 / SQRT_3 / 2.0, 0.25),
                (1.5 / SQRT_3, -1.0 / 2.0),
                (0.0, -1.0 / 2.0),
                ORIGIN,
                (0.0, -1.0 / 2.0),
                (-1.5 / SQRT_3, -1.0 / 2.0),
                (-1.5 / SQRT_3 / 2.0, 0.25),
                ORIGIN,
                (-1.5 / SQRT_3 / 2.0, 0.25),
            ]
            .into(),
        )
    }
}

impl<'a, Coord, Size: SizeDesc> PointCollection<'a, Coord> for &'a PathMarker<Coord, Size> {
    type IntoIter = std::iter::Once<&'a Coord>;
    type Point = &'a Coord;

    fn point_iter(self) -> std::iter::Once<&'a Coord> {
        std::iter::once(&self.coord)
    }
}
impl<DB: DrawingBackend, Coord, Size: SizeDesc> Drawable<DB> for PathMarker<Coord, Size> {
    fn draw<I: Iterator<Item = BackendCoord>>(
        &self,
        mut pos: I,
        backend: &mut DB,
        parent_dim: (u32, u32),
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        let Some((ax, ay)) = pos.next() else {
            return Ok(());
        };
        let size = self.size.in_pixels(&parent_dim).max(0) as u32;
        let points: Vec<BackendCoord> = self
            .path
            .to_size(size) // ← ownership issue here
            .into_iter()
            .map(|(x, y)| (ax + x, ay + y))
            .collect();

        if let Some(style) = &self.fill {
            backend.fill_polygon(points.clone(), style)?;
        }
        if let Some(style) = &self.stroke {
            let mut points = points;
            points.extend_from_within(0..2);
            backend.draw_path(points, style)?;
        }
        Ok(())
    }
}

fn build_demo_graph<P: AsRef<Path>>(output_dir: &P) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    let plot_path = output_dir.join("demo.svg");
    log::info!("Plotting to {}", plot_path.display());

    let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("demo", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(0..10, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Index")
        .y_desc("Value")
        .y_label_formatter(&|&bps| {
            format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
        })
        .draw()?;

    let size = 10;

    for (i, (stroke, fill)) in [
        (colors::RED.stroke_width(2), None),
        (colors::BLACK.stroke_width(2), colors::RED.filled().into()),
    ]
    .iter()
    .enumerate()
    {
        let idx = i as i32 + 1;
        let stroke: ShapeStyle = *stroke;
        let fill: Option<ShapeStyle> = *fill;

        chart.draw_series([(idx, 0.10)].map(|coord| Circle::new(coord, size, stroke)))?;

        chart.draw_series(
            [(idx, 0.20)].map(|coord| PathMarker::square(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 0.30)].map(|coord| PathMarker::diamond(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 0.40)].map(|coord| PathMarker::tri_up(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 0.50)].map(|coord| PathMarker::tri_down(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 0.60)].map(|coord| PathMarker::cross_square(coord, size, stroke, fill)),
        )?;
        chart
            .draw_series([(idx, 0.70)].map(|coord| PathMarker::star(coord, size, stroke, fill)))?;
        chart.draw_series(
            [(idx, 0.80)].map(|coord| PathMarker::cross_diamond(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 0.90)].map(|coord| PathMarker::cross_tri_up(coord, size, stroke, fill)),
        )?;
        chart.draw_series(
            [(idx, 1.0)].map(|coord| PathMarker::cross_tri_down(coord, size, stroke, fill)),
        )?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn build_internal_rel_tgraph<P: AsRef<Path>>(
    model: &str,
    accel: bool,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let span_map: BTreeMap<&str, ShapeStyle> = [
        ("bpe_backtrack", colors::GREY_A400.filled()),
        ("buffer_sweep", colors::GREEN_A400.filled()),
        ("merge_heap", colors::BLUE_A400.filled()),
        ("priority_merge", colors::PINK_A700.filled()),
        ("tail_sweep", colors::DEEPPURPLE_A400.filled()),
    ]
    .iter()
    .cloned()
    .collect();

    let span_key = |span: &str| {
        format!(
            "encoding_parallel::wordchipper::{span}::{model}{}",
            if accel { "_fast" } else { "" },
        )
    };

    let mut plot_series: Vec<Series> = Default::default();
    for (span, style) in span_map.iter() {
        if let Some(series_data) = data.select_series(&span_key(span)) {
            let style = if accel { style.filled() } else { *style };

            plot_series.push(Series {
                name: span.to_string(),
                glyph: "X".to_string(),
                style,
                points: as_points(&series_data, |_, br| median_bps(br)),
            })
        }
    }

    // Normalize the points to the max value.
    let mut baseline: BTreeMap<u32, f64> = Default::default();
    for series in plot_series.iter() {
        for point in series.points.iter() {
            let entry = baseline.entry(point.threads).or_default();
            *entry = fmax(*entry, point.value);
        }
    }
    for series in plot_series.iter_mut() {
        series.points.iter_mut().for_each(|point| {
            point.value /= baseline[&point.threads];
        })
    }

    let min_threads = plot_series.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = plot_series.iter().map(|s| s.max_threads()).max().unwrap();
    let min_bps = plot_series
        .iter()
        .map(|s| s.min_bps())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_bps = plot_series
        .iter()
        .map(|s| s.max_bps())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "encoder vrs max, {} lexer, model: \"{}\"",
                if accel { "logos" } else { "regex" },
                model
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Relative Median Throughput")
        .draw()?;

    for pseries in plot_series {
        let name = &pseries.name;
        let style = pseries.style;
        let points: Vec<(u32, f64)> = pseries
            .points
            .iter()
            .map(|p| (p.threads, p.value))
            .collect();

        let size = 4;
        chart
            .draw_series(
                points
                    .iter()
                    .map(|coord| EmptyElement::at(*coord) + Circle::new((0, 0), size, style)),
            )?
            .label(name)
            .legend(move |coord| EmptyElement::at(coord) + Circle::new((0, 0), size, style));

        chart.draw_series(LineSeries::new(
            pseries.points.iter().map(|p| (p.threads, p.value)),
            style,
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
fn build_internal_tgraph<P: AsRef<Path>>(
    model: &str,
    accel: bool,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let span_map: BTreeMap<&str, ShapeStyle> = [
        ("bpe_backtrack", colors::GREY_A400.filled()),
        ("buffer_sweep", colors::GREEN_A400.filled()),
        ("merge_heap", colors::BLUE_A400.filled()),
        ("priority_merge", colors::PINK_A700.filled()),
        ("tail_sweep", colors::DEEPPURPLE_A400.filled()),
    ]
    .iter()
    .cloned()
    .collect();

    let mut plot_series: Vec<Series> = Default::default();

    for (span, style) in span_map.iter() {
        let sname = format!(
            "encoding_parallel::wordchipper::{span}::{model}{}",
            if accel { "_fast" } else { "" },
        );

        if let Some(series_data) = data.select_series(&sname) {
            let style = if accel { style.filled() } else { *style };

            plot_series.push(Series {
                name: span.to_string(),
                glyph: "X".to_string(),
                style,
                points: as_points(&series_data, |_, br| median_bps(br)),
            })
        }
    }

    let min_threads = plot_series.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = plot_series.iter().map(|s| s.max_threads()).max().unwrap();
    let min_bps = plot_series
        .iter()
        .map(|s| s.min_bps())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_bps = plot_series
        .iter()
        .map(|s| s.max_bps())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "encoder vrs, {} lexer, model: \"{}\"",
                if accel { "logos" } else { "regex" },
                model
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Median Throughput")
        .y_label_formatter(&|&bps| {
            format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
        })
        .draw()?;

    for pseries in plot_series {
        let name = &pseries.name;
        let style = pseries.style;
        let points: Vec<(u32, f64)> = pseries
            .points
            .iter()
            .map(|p| (p.threads, p.value))
            .collect();

        let size = 4;
        chart
            .draw_series(
                points
                    .iter()
                    .map(|coord| EmptyElement::at(*coord) + Circle::new((0, 0), size, style)),
            )?
            .label(name)
            .legend(move |coord| EmptyElement::at(coord) + Circle::new((0, 0), size, style));

        chart.draw_series(LineSeries::new(
            pseries.points.iter().map(|p| (p.threads, p.value)),
            style,
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
fn build_external_tgraph<P: AsRef<Path>>(
    model: &str,
    span_encoder: &str,
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let series_names = data
        .series_names()
        .into_iter()
        .filter(|name| name.contains(model))
        .collect::<Vec<_>>();

    let mut brandx_group: Vec<Series> = Default::default();
    for (ext, (glyph, style)) in external_styles().iter() {
        if let Some(name) = series_names.iter().find(|name| name.contains(ext)) {
            let series_data = data.select_series(name).unwrap();

            brandx_group.push(Series {
                name: ext.to_string(),
                glyph: glyph.to_string(),
                style: *style,
                points: as_points(&series_data, |_, br| median_bps(br)),
            })
        }
    }

    let regex_series = Series {
        name: "wordchipper:regex".to_string(),
        glyph: "✦".to_string(),
        style: colors::GREEN_200.filled(),
        points: as_points(
            &data
                .select_series(&format!(
                    "encoding_parallel::wordchipper::{span_encoder}::{model}"
                ))
                .expect("Failed to select regex series"),
            |_, br| median_bps(br),
        ),
    };

    let logos_series = Series {
        name: "wordchipper:logos".to_string(),
        glyph: "★".to_string(),
        style: colors::LIGHTBLUE_200.filled(),
        points: as_points(
            &data
                .select_series(&format!(
                    "encoding_parallel::wordchipper::{span_encoder}::{model}_fast"
                ))
                .expect("Failed to select regex series"),
            |_, br| median_bps(br),
        ),
    };

    let min_threads = brandx_group.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = brandx_group.iter().map(|s| s.max_threads()).max().unwrap();

    let min_bps = fmin(
        brandx_group
            .iter()
            .map(|s| s.min_bps())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        regex_series.min_bps(),
    );
    let max_bps = fmax(
        brandx_group
            .iter()
            .map(|s| s.max_bps())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        regex_series.max_bps(),
    );

    for include_logos in [false, true] {
        let chart_name = if include_logos { "logos" } else { "regex" };

        let plot_path = output_dir.join(format!("wc_{chart_name}_vrs_brandx.rust.{model}.svg"));
        log::info!("Plotting to {}", plot_path.display());

        let min_bps = if include_logos {
            fmin(min_bps, logos_series.min_bps())
        } else {
            min_bps
        };
        let max_bps = if include_logos {
            fmax(max_bps, logos_series.max_bps())
        } else {
            max_bps
        };

        let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
        root.fill(&colors::WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("wordchipper:{chart_name} vrs brandx, rust, model: \"{model}\"",),
                ("sans-serif", 20).into_font(),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(90)
            .build_cartesian_2d(
                (min_threads..max_threads).log_scale().base(2.0),
                (min_bps..max_bps).log_scale().base(2.0),
            )?;

        chart
            .configure_mesh()
            .x_desc("Thread Count")
            .y_desc("Median Throughput")
            .y_label_formatter(&|&bps| {
                format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
            })
            .draw()?;

        let glyph_size = 24;

        for pseries in brandx_group.iter() {
            let name = &pseries.name;
            let style = pseries.style;
            let points: Vec<(u32, f64)> = pseries
                .points
                .iter()
                .map(|p| (p.threads, p.value))
                .collect();

            chart.draw_series(LineSeries::new(
                pseries.points.iter().map(|p| (p.threads, p.value)),
                style.stroke_width(4),
            ))?;

            chart
                .draw_series(points.iter().map(|coord| {
                    EmptyElement::at(*coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                }))?
                .label(name)
                .legend(move |coord| {
                    let glyph_size = glyph_size * 2 / 3;

                    EmptyElement::at(coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                });
        }

        for pseries in if include_logos {
            vec![&logos_series, &regex_series]
        } else {
            vec![&regex_series]
        } {
            let name = &pseries.name;
            let style = pseries.style;
            let points: Vec<(u32, f64)> = pseries
                .points
                .iter()
                .map(|p| (p.threads, p.value))
                .collect();

            chart.draw_series(LineSeries::new(
                pseries.points.iter().map(|p| (p.threads, p.value)),
                style.stroke_width(4),
            ))?;

            chart
                .draw_series(points.iter().map(|coord| {
                    EmptyElement::at(*coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                }))?
                .label(name)
                .legend(move |coord| {
                    let glyph_size = glyph_size * 2 / 3;

                    EmptyElement::at(coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}
