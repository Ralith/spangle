//! Generate plausible, aesthetically appealing spherical starfields

use std::array;

use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_distr::Uniform;
use rand_pcg::Pcg64Mcg;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::Simd;
use simdnoise::simplex::simplex_3d;

/// Parameters defining a starfield
#[derive(Debug, Clone)]
pub struct Starfield {
    count: usize,
    seed: [u8; 16],
    irradiance_frequency: f32,
    expected_mean_irradiance: f32,
}

impl Starfield {
    pub fn new() -> Self {
        Self {
            count: 20_000,
            irradiance_frequency: 10.0,
            expected_mean_irradiance: 1.0,
            seed: [
                0x1c, 0x3d, 0xaa, 0x82, 0xd2, 0x17, 0xf3, 0xe6, 0xa4, 0xf8, 0x87, 0xfa, 0x91, 0x37,
                0x39, 0x1a,
            ],
        }
    }

    /// Number of stars
    ///
    /// Note that viewers on the ground will only about half of them.
    pub fn count(&mut self, value: usize) -> &mut Self {
        self.count = value;
        self
    }

    /// Random number generator seed
    pub fn seed(&mut self, value: [u8; 16]) -> &mut Self {
        self.seed = value;
        self
    }

    /// Unitless scaling factor for the angular rate at which irradiance varies
    ///
    /// Values between 1 and 10 are a good place to start.
    pub fn irradiance_frequency(&mut self, value: f32) -> &mut Self {
        self.irradiance_frequency = value;
        self
    }

    /// W/m^2 received from the average star
    ///
    /// We work in terms of irradiance rather than luminance to ensure that redder stars appear
    /// dimmer naturally.
    pub fn expected_mean_irradiance(&mut self, value: f32) -> &mut Self {
        self.expected_mean_irradiance = value;
        self
    }

    pub fn generate(&self) -> StarIter {
        let mut rng = Pcg64Mcg::from_seed(self.seed);
        let mut block = [([0.0; 3], 0.0); BLOCK_SIZE - 1].into_iter();
        block.nth(BLOCK_SIZE - 1); // Make block empty
        StarIter {
            noise_seed: rng.gen(),
            rng,
            irradiance_frequency: self.irradiance_frequency,
            // * 2 to account for the simplex attenuation, * 2 to account for the uniform
            // distribution
            irradiance: Uniform::new(0.0, self.expected_mean_irradiance * 2.0 * 2.0),
            remaining: self.count,
            block,
        }
    }
}

impl Default for Starfield {
    fn default() -> Self {
        Self::new()
    }
}

/// Properties of a single star in a starfield
#[derive(Debug, Copy, Clone)]
pub struct Star {
    /// Unit vector encoding the direction of the star
    pub direction: [f32; 3],
    /// W/m^2 received from this star
    pub irradiance: f32,
    /// Kelvin, suitable for deriving a color from according to black-body radiation
    pub temperature: f32,
}

pub struct StarIter {
    rng: Pcg64Mcg,
    noise_seed: i32,
    irradiance_frequency: f32,
    irradiance: Uniform<f32>,
    remaining: usize,
    block: array::IntoIter<([f32; 3], f32), { BLOCK_SIZE - 1 }>,
}

impl Iterator for StarIter {
    type Item = Star;
    fn next(&mut self) -> Option<Star> {
        self.remaining = self.remaining.checked_sub(1)?;

        let (direction, irradiance) = match self.block.next() {
            Some(x) => x,
            None => {
                let mut x = Block([(); BLOCK_SIZE].map(|()| self.rng.sample(StandardNormal)));
                let mut y = Block([(); BLOCK_SIZE].map(|()| self.rng.sample(StandardNormal)));
                let mut z = Block([(); BLOCK_SIZE].map(|()| self.rng.sample(StandardNormal)));
                let irradiance =
                    Block([(); BLOCK_SIZE].map(|()| self.rng.sample(&self.irradiance)));
                let samples = normalize_and_sample_runtime_select(
                    self.noise_seed,
                    self.irradiance_frequency,
                    &irradiance,
                    &mut x,
                    &mut y,
                    &mut z,
                );
                let mut i = 0;
                self.block = [(); BLOCK_SIZE - 1]
                    .map(|()| {
                        i += 1;
                        ([x.0[i], y.0[i], z.0[i]], samples.0[i])
                    })
                    .into_iter();
                ([x.0[0], y.0[0], z.0[0]], samples.0[0])
            }
        };

        let temperature = self.rng.sample::<f32, _>(rand_distr::Exp1) * 10_000.0 + 2_000.0;

        Some(Star {
            direction,
            irradiance,
            temperature,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StarIter {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Computes the linear sRGB color with the same hue as a black body at a temperature of `kelvin`
///
/// To obtain scene-referred irradiance, scale the result such that the sum of the channels equal
/// the desired total irradiance.
pub fn black_body_color(kelvin: f32) -> [f32; 3] {
    let [x, y] = kelvin_to_xy(kelvin);
    // Y = 0.0722 is the highest relative luminance in the sRGB gamut for which no in-gamut
    // chromaticity will produce clipping, aka the luminance of the sRGB (0, 0, 1), because blue is
    // the least luminous sRGB primary.
    xyz_to_linear_srgb(xyy_to_xyz([x, y, 0.0722]))
}

/// Kelvin to CIE 1931 chromaticity coordinates
fn kelvin_to_xy(t: f32) -> [f32; 2] {
    // https://en.wikipedia.org/wiki/Planckian_locus
    let t = t.clamp(1_667.0, 25_000.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039
    };
    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483
    };
    [x, y]
}

/// CIE 1931 xyY chromaticity coordinates to CIE 1931 XYZ
fn xyy_to_xyz([cx, cy, y]: [f32; 3]) -> [f32; 3] {
    let x = cx * y / cy;
    let z = (1.0 - cx - cy) * y / cy;
    [x, y, z]
}

fn xyz_to_linear_srgb([x, y, z]: [f32; 3]) -> [f32; 3] {
    // https://color.org/chardata/rgb/sRGB.pdf
    [
        3.2406255 * x - 1.537208 * y - 0.4986286 * z,
        -0.9689307 * x + 1.8757561 * y + 0.0415175 * z,
        0.0557101 * x - 0.2040211 * y + 1.0569959 * z,
    ]
    .map(|c| c.clamp(0.0, 1.0))
}

#[derive(Debug)]
#[repr(align(512))]
struct Block([f32; BLOCK_SIZE]);

const BLOCK_SIZE: usize = 128;

simd_runtime_generate!(
    fn normalize_and_sample(
        seed: i32,
        frequency: f32,
        irradiance: &Block,
        x: &mut Block,
        y: &mut Block,
        z: &mut Block,
    ) -> Block {
        assert!(BLOCK_SIZE >= S::VF32_WIDTH);
        let mut out = Block([0.0; BLOCK_SIZE]);
        for ((((x, y), z), irradiance), out) in
            x.0.chunks_mut(S::VF32_WIDTH)
                .zip(y.0.chunks_mut(S::VF32_WIDTH))
                .zip(z.0.chunks_mut(S::VF32_WIDTH))
                .zip(irradiance.0.chunks(S::VF32_WIDTH))
                .zip(out.0.chunks_mut(S::VF32_WIDTH))
        {
            // Normalize
            let x_reg = S::load_ps(&x[0]);
            let y_reg = S::load_ps(&y[0]);
            let z_reg = S::load_ps(&z[0]);
            let inverse_len =
                S::set1_ps(1.0) / S::sqrt_ps(x_reg * x_reg + y_reg * y_reg + z_reg * z_reg);
            let x_unit = x_reg * inverse_len;
            let y_unit = y_reg * inverse_len;
            let z_unit = z_reg * inverse_len;
            S::store_ps(&mut x[0], x_unit);
            S::store_ps(&mut y[0], y_unit);
            S::store_ps(&mut z[0], z_unit);

            // Sample
            let freq = S::set1_ps(frequency);
            let x_scaled = freq * x_unit;
            let y_scaled = freq * y_unit;
            let z_scaled = freq * z_unit;
            let samples = simplex_3d::<S>(x_scaled, y_scaled, z_scaled, seed);
            let mapped = S::fmadd_ps(S::set1_ps(0.5), samples, S::set1_ps(0.5));
            S::store_ps(&mut out[0], mapped * S::load_ps(&irradiance[0]))
        }
        out
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_sanity() {
        let seed = rand::thread_rng().gen();
        println!("seed: {:x?}", seed);
        let iter = Starfield::new().seed(seed).generate().take(100);
        for star in iter {
            let len = star.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((len - 1.0).abs() < 1e-3, "direction has unit length");
            assert!(star.irradiance > 0.0, "irradiance is positive");
            assert!(star.temperature > 0.0, "temperature is positive");
        }
    }

    #[test]
    fn black_body_color_sanity() {
        let low = black_body_color(2000.0);
        // Low temperatures are reddish
        assert!(low[0] > low[1] && low[0] > low[2]);
        let high = black_body_color(40000.0);
        // Low temperatures are blueish
        assert!(high[2] > high[1] && high[2] > high[0]);
    }
}
