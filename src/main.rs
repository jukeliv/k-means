use plotters::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: f32,
    y: f32
}

#[derive(Clone, Copy, PartialEq)]
struct Point3 {
    x: f32,
    y: f32,
    z: f32
}

trait CalculateCentroid<T> {
    fn calculate_centroid(cluster: &[T]) -> T;
}

/*
 *  approach: https://en.wikipedia.org/wiki/Xorshift
 *  IN MY DEFENSE! I was only allowed to use a plotting library for this challenge of implementing K-Means
 *  and I like the C API for PRNG, that's my defence for this code.
*/
static mut RANDOM_STATE: usize = 69420;

fn rand() -> usize {
    unsafe {
        let mut x = RANDOM_STATE;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        RANDOM_STATE = x;
        RANDOM_STATE
    }
}

fn srand(seed: usize) {
    unsafe {
        RANDOM_STATE = seed;
    }
}

fn rand_float() -> f32 {
    rand() as f32 / usize::MAX as f32
}

fn argmin<T: std::cmp::PartialOrd + Copy>(elements: Vec<T>) -> Option<usize> {
    if elements.len() == 0 {
        return None;
    }

    let mut min = elements[0];
    let mut index = 0;
    for (i, element) in elements.iter().enumerate() {
        if *element < min {
            min = *element;
            index = i;
        }
    }

    Some(index)
}

// the comments in this function come from the wikipedia article where I "borrowed" the pseudocode. (https://en.wikipedia.org/wiki/K-means_clustering#Variations)
fn k_means_cluster<T: PartialEq + Copy + CalculateCentroid<T>>(k: usize, points: Vec<T>, dist: fn(a: T, b: T) -> f32, max_iterations: usize) -> Vec<Vec<T>> {
    // Initialization: choose k centroids (Forgy, Random Partition, etc.)
    // centroids = [c1, c2, ..., ck]
    let mut centroids = Vec::new();
    for _ in 0..k {
        let idx = rand() % points.len();
        centroids.push(points[idx]);
    }

    // Initialize clusters list
    let mut clusters: Vec<Vec<T>> = Vec::new();
    
    for _ in 0.. max_iterations {
        // Clear previous clusters
        clusters = Vec::with_capacity(k);
        for _ in 0..k {
            clusters.push(vec![]);
        }
    
        // Assign each point to the "closest" centroid 
        for point in &points {
            let mut distances_to_each_centroid = Vec::new();
            for centroid in &centroids {
                distances_to_each_centroid.push(dist(*point, *centroid));
            }
            
            let cluster_assignment = argmin(distances_to_each_centroid).unwrap();
            clusters[cluster_assignment].push(*point)
        }
        
        // Calculate new centroids
        //   (the standard implementation uses the mean of all points in a
        //     cluster to determine the new centroid)
        let mut new_centroids = Vec::new();
        for cluster in clusters.iter() {
            if cluster.is_empty() {
                new_centroids.push(points[rand() % points.len()]);
            } else {
                new_centroids.push(T::calculate_centroid(&cluster));
            }
        }

        let mut converged = true;
        for (c1, c2) in new_centroids.iter().zip(&centroids) {
            if dist(*c1, *c2) > 1e-4 {
                converged = false;
                break;
            }
        }
        
        if converged {
            break;
        }

        centroids = new_centroids;
    }

    return clusters;
}

impl CalculateCentroid<Point> for Point {
    fn calculate_centroid(cluster: &[Point]) -> Point {
        let mut mean:Point = Point{
            x: 0.0,
            y: 0.0
        };
        for var in cluster {
            mean.x += var.x;
            mean.y += var.y;
        }
        mean.x /= cluster.len() as f32;
        mean.y /= cluster.len() as f32;

        mean
    }
}

impl CalculateCentroid<Point3> for Point3 {
    fn calculate_centroid(cluster: &[Point3]) -> Point3 {
        let mut mean:Point3 = Point3 {x: 0.0, y: 0.0, z: 0.0};
        for var in cluster {
            mean.x += var.x;
            mean.y += var.y;
            mean.z += var.z;
        }
        mean.x /= cluster.len() as f32;
        mean.y /= cluster.len() as f32;
        mean.z /= cluster.len() as f32;

        mean
    }
}

fn generate_circle(data: &mut Vec<Point>, max_points: u64, radius: f32, center: Point) {
    for _ in 0.. max_points {
        let angle = rand_float() * 2.0 * std::f32::consts::PI;
        let mag = rand_float();
        let p = Point {
            x: center.x + angle.sin() * mag * radius as f32,
            y: center.y + angle.cos() * mag * radius as f32,
        };
        data.push(p)
    }
}

fn generate_sphere(data: &mut Vec<Point3>, max_points: u64, radius: f32, center: Point3) {
    for _ in 0.. max_points {
        let u = rand_float();
        let v = rand_float();
        let theta = 2.0 * std::f32::consts::PI * u;
        let phi = (2.0 * v - 1.0).acos();

        data.push(Point3 {
            x: center.x + (radius * phi.sin() * theta.cos()),
            y: center.y + (radius * phi.sin() * theta.sin()),
            z: center.z + (radius * phi.cos())
        });
    }
}

fn main() {
    let args:Vec<String> = std::env::args().collect();
    srand(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as usize);

    let points = 1e6 as u64;
    let k = 3;
    
    if args.len() > 1 && args[1] == "-sphere" {
        let mut data = Vec::new();

        generate_sphere(&mut data, points / 3, 20.0, Point3 { x: 0.0, y: -1.7 * 20.0, z: 1.2 * 20.0 });
        generate_sphere(&mut data, points / 3, 32.0, Point3 { x: 0.0, y: 0.0, z:  0.0 });
        generate_sphere(&mut data, points / 3, 20.0, Point3 { x: 0.0, y: 1.7 * 20.0, z: 1.2 * 20.0 });

        let start = std::time::Instant::now();
        let clusters = k_means_cluster(k, data, |a, b| (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2), 1000);
        println!("took {} ms for ~{points} points", start.elapsed().as_millis());
        
        // Setup 3D plot area
        let backend = BitMapBackend::new("clusters_3d.png", (1024, 768));
        let root = backend.into_drawing_area();
        root.fill(&WHITE).expect("couldn't fill root");

        // Compute min/max for each axis
        let min_x = clusters.iter().flatten().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let max_x = clusters.iter().flatten().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = clusters.iter().flatten().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_y = clusters.iter().flatten().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let min_z = clusters.iter().flatten().map(|p| p.z).fold(f32::INFINITY, f32::min);
        let max_z = clusters.iter().flatten().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("3D K‑Means Clusters", ("sans-serif", 30))
            .margin(20)
            .build_cartesian_3d(min_x..max_x, min_y..max_y, min_z..max_z).expect("couldn't build cartesian plane");

        chart.with_projection(|mut pb| {
            pb.yaw = 0.8;
            pb.pitch = 0.3;
            pb.scale = 0.9;
            pb.into_matrix()
        });

        chart.configure_axes().draw().expect("couldn't draw cartesian plane");

        let colors = [
            &GREEN, &CYAN, &YELLOW, &BLACK,
            &RGBColor(255, 128, 0),
            &RGBColor(128, 0, 255),
            &RGBColor(0, 128, 255),
        ];

        for (i, cluster) in clusters.iter().enumerate() {
            let color = colors[i % colors.len()];
            chart.draw_series(
                cluster.iter().map(|p| Cubiod::new([(p.x, p.y, p.z), ((p.x) + 1.0, (p.y) + 1.0, (p.z) + 1.0,)], color.mix(0.2), color.filled(),
            ))).expect("couldn't draw points");
        }

        // Draw centroids with larger spheres
        for cluster in clusters.iter() {
            let centroid = Point3::calculate_centroid(cluster);
            chart.draw_series(std::iter::once(
                Cubiod::new([(centroid.x, centroid.y, centroid.z), (centroid.x + 1.0, centroid.y+ 1.0, centroid.z+ 1.0)], BLACK.mix(0.8), BLACK.filled()),
            )).expect("couldn't draw centroids");
        }

        root.present().expect("couldn't draw plot");
        println!("Saved 3D scatter to clusters_3d.png");
    } else {
        let mut data = Vec::new();

        generate_circle(&mut data, points/3, 10.0, Point {x: 20.0, y : 20.0});
        generate_circle(&mut data, points/3, 20.0, Point {x: 40.0, y : 00.0});
        generate_circle(&mut data, points/3, 10.0, Point {x: 60.0, y : 20.0});

        let start = std::time::Instant::now();
        let clusters = k_means_cluster(k, data, |a, b| (a.x - b.x).powi(2) + (a.y - b.y).powi(2), 1000);
        println!("took {} ms for ~{points} points", start.elapsed().as_millis());
        
        // IDK how tf this works, why is it done this way nor anything, this is pure ChatGPT magic tbh.
        let min_x = clusters.iter().flatten().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let max_x = clusters.iter().flatten().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = clusters.iter().flatten().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_y = clusters.iter().flatten().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

        let root = BitMapBackend::new("clusters.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE).expect("couldn't fill root");

        let mut chart = ChartBuilder::on(&root)
            .caption("Clusters of Points", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(min_x..max_x, min_y..max_y).expect("couldn't generate chart");

        chart.configure_mesh().draw().expect("couldn't plot cartesian plane");

        let colors = [
            &GREEN, &CYAN, &YELLOW, &BLACK,
            &RGBColor(255, 128, 0),
            &RGBColor(128, 0, 255),
            &RGBColor(0, 128, 255),
        ];

        for (i, cluster) in clusters.iter().enumerate() {
            let color = colors[i % colors.len()];
            chart.draw_series(
                cluster.iter().map(|p| Circle::new((p.x, p.y), 5, color.filled())),
            ).expect("couldn't plot points");

            let centroid = Point::calculate_centroid(cluster);

            chart.draw_series(std::iter::once(
                Cross::new((centroid.x, centroid.y), 10, &RED),
            )).expect("couldn't plot centroids");
        }

        root.present().expect("couldn't present plot");
        println!("Saved scatter to clusters.png");
    }
}