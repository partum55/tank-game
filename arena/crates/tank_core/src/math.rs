use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, Neg};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        (self - other).length_squared()
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-6 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn from_angle(radians: f32) -> Self {
        Self {
            x: radians.cos(),
            y: radians.sin(),
        }
    }

    #[inline]
    pub fn to_angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    #[inline]
    pub fn rotate(self, radians: f32) -> Self {
        let cos = radians.cos();
        let sin = radians.sin();
        Self {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs }
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self { x: self.x / rhs, y: self.y / rhs }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec2,
    pub dir: Vec2,
}

/// Ray vs Circle intersection.
/// Returns Some((distance_t, hit_point)) if ray hits circle with t >= 0.
pub fn ray_circle_intersect(origin: Vec2, dir: Vec2, center: Vec2, radius: f32) -> Option<(f32, Vec2)> {
    let d = dir.normalize();
    let m = origin - center;
    let b = m.dot(d);
    let c = m.dot(m) - radius * radius;

    // Ray origin outside circle and pointing away
    if c > 0.0 && b > 0.0 {
        return None;
    }

    let discr = b * b - c;
    if discr < 0.0 {
        return None;
    }

    let sqrt_discr = discr.sqrt();
    let mut t = -b - sqrt_discr;
    if t < 0.0 {
        t = -b + sqrt_discr;
    }

    if t < 0.0 {
        return None;
    }

    let hit_point = origin + d * t;
    Some((t, hit_point))
}

/// Ray vs AABB / Rectangle bounds intersection (for walls).
/// Returns Some(t) where t is distance along ray.
pub fn ray_bounds_intersect(origin: Vec2, dir: Vec2, width: f32, height: f32) -> Option<f32> {
    let mut t_min = 0.0f32;
    let mut t_max = f32::INFINITY;

    // X axis
    if dir.x.abs() > 1e-6 {
        let t1 = (0.0 - origin.x) / dir.x;
        let t2 = (width - origin.x) / dir.x;
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);
        if t_min > t_max || t_max < 0.0 {
            return None;
        }
    }

    // Y axis
    if dir.y.abs() > 1e-6 {
        let t1 = (0.0 - origin.y) / dir.y;
        let t2 = (height - origin.y) / dir.y;
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);
        if t_min > t_max || t_max < 0.0 {
            return None;
        }
    }

    if t_min > 0.0 {
        Some(t_min)
    } else if t_max > 0.0 {
        Some(t_max)
    } else {
        None
    }
}

/// Ray vs Axis-Aligned Bounding Box (AABB) intersection.
/// Returns Some(t) where t >= 0 is distance along ray to intersection.
pub fn ray_rect_intersect(
    origin: Vec2,
    dir: Vec2,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
) -> Option<f32> {
    let mut t_min = 0.0f32;
    let mut t_max = f32::INFINITY;

    let x_min = rx;
    let x_max = rx + rw;
    let y_min = ry;
    let y_max = ry + rh;

    // Check X slab
    if dir.x.abs() > 1e-6 {
        let t1 = (x_min - origin.x) / dir.x;
        let t2 = (x_max - origin.x) / dir.x;
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);
        if t_min > t_max || t_max < 0.0 {
            return None;
        }
    } else if origin.x < x_min || origin.x > x_max {
        return None;
    }

    // Check Y slab
    if dir.y.abs() > 1e-6 {
        let t1 = (y_min - origin.y) / dir.y;
        let t2 = (y_max - origin.y) / dir.y;
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(t_near);
        t_max = t_max.min(t_far);
        if t_min > t_max || t_max < 0.0 {
            return None;
        }
    } else if origin.y < y_min || origin.y > y_max {
        return None;
    }

    if t_min >= 0.0 {
        Some(t_min)
    } else if t_max >= 0.0 {
        Some(t_max)
    } else {
        None
    }
}

/// Checks if line segment between p1 and p2 intersects the rectangle.
pub fn segment_rect_intersect(
    p1: Vec2,
    p2: Vec2,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
) -> bool {
    let d = p2 - p1;
    let len = d.length();
    if len < 1e-6 {
        return false;
    }
    let dir = d / len;
    if let Some(t) = ray_rect_intersect(p1, dir, rx, ry, rw, rh) {
        t <= len
    } else {
        false
    }
}

/// Resolves circle vs AABB rectangle collision by pushing the circle out of the rectangle.
/// Returns true if a collision was detected and resolved.
pub fn circle_rect_resolve(
    circle_pos: &mut Vec2,
    radius: f32,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
) -> bool {
    let closest_x = circle_pos.x.clamp(rx, rx + rw);
    let closest_y = circle_pos.y.clamp(ry, ry + rh);

    let diff = *circle_pos - Vec2::new(closest_x, closest_y);
    let dist_sq = diff.length_squared();

    if dist_sq < radius * radius && dist_sq > 1e-6 {
        let dist = dist_sq.sqrt();
        let overlap = radius - dist;
        let normal = diff / dist;
        *circle_pos += normal * overlap;
        true
    } else if dist_sq <= 1e-6 {
        // Circle center is inside rectangle: push out to nearest edge
        let d_left = (circle_pos.x - rx).abs();
        let d_right = (rx + rw - circle_pos.x).abs();
        let d_top = (circle_pos.y - ry).abs();
        let d_bottom = (ry + rh - circle_pos.y).abs();

        let min_d = d_left.min(d_right).min(d_top).min(d_bottom);
        if (min_d - d_left).abs() < 1e-5 {
            circle_pos.x = rx - radius;
        } else if (min_d - d_right).abs() < 1e-5 {
            circle_pos.x = rx + rw + radius;
        } else if (min_d - d_top).abs() < 1e-5 {
            circle_pos.y = ry - radius;
        } else {
            circle_pos.y = ry + rh + radius;
        }
        true
    } else {
        false
    }
}

/// Point `p` expressed in the frame of a box centred at `center`, heading `angle` (x = along the heading).
#[inline]
pub fn to_local(p: Vec2, center: Vec2, angle: f32) -> Vec2 {
    let d = p - center;
    let (s, c) = angle.sin_cos();
    Vec2::new(d.x * c + d.y * s, -d.x * s + d.y * c)
}

/// Ray vs oriented box (`half_len` along `angle`, `half_wid` across). Same conventions as `ray_rect_intersect`.
pub fn ray_obb_intersect(
    origin: Vec2,
    dir: Vec2,
    center: Vec2,
    angle: f32,
    half_len: f32,
    half_wid: f32,
) -> Option<f32> {
    let o = to_local(origin, center, angle);
    let (s, c) = angle.sin_cos();
    let d = Vec2::new(dir.x * c + dir.y * s, -dir.x * s + dir.y * c);
    ray_rect_intersect(o, d, -half_len, -half_wid, 2.0 * half_len, 2.0 * half_wid)
}

/// Axis-aligned half extents of an oriented box (for keeping it inside the arena).
#[inline]
pub fn obb_half_extents(angle: f32, half_len: f32, half_wid: f32) -> (f32, f32) {
    let (s, c) = angle.sin_cos();
    (c.abs() * half_len + s.abs() * half_wid, s.abs() * half_len + c.abs() * half_wid)
}

/// SAT between two oriented boxes. `Some(v)` = the translation that moves box 1 out of box 2 with the
/// smallest push (move box 2 by `-v` instead, or split). Boxes: (centre, angle, half_len, half_wid).
pub fn obb_obb_mtv(
    c1: Vec2,
    a1: f32,
    h1: (f32, f32),
    c2: Vec2,
    a2: f32,
    h2: (f32, f32),
) -> Option<Vec2> {
    // axes written as (cos, sin) / (-sin, cos) so that the GPU copy (gpu_arena.py) picks the same axis on ties
    let (s1, k1) = a1.sin_cos();
    let (s2, k2) = a2.sin_cos();
    let (u1, v1) = (Vec2::new(k1, s1), Vec2::new(-s1, k1));
    let (u2, v2) = (Vec2::new(k2, s2), Vec2::new(-s2, k2));
    let delta = c1 - c2;
    let mut best = f32::INFINITY;
    let mut push = Vec2::ZERO;
    for axis in [u1, v1, u2, v2] {
        let r1 = (h1.0 * u1.dot(axis)).abs() + (h1.1 * v1.dot(axis)).abs();
        let r2 = (h2.0 * u2.dot(axis)).abs() + (h2.1 * v2.dot(axis)).abs();
        let dist = delta.dot(axis);
        let overlap = r1 + r2 - dist.abs();
        if overlap <= 0.0 {
            return None;
        }
        if overlap < best {
            best = overlap;
            push = axis * (if dist >= 0.0 { overlap } else { -overlap });
        }
    }
    Some(push)
}

/// SAT of an oriented box against an axis-aligned rectangle (a wall). Same result convention as `obb_obb_mtv`.
pub fn obb_rect_mtv(
    center: Vec2,
    angle: f32,
    half: (f32, f32),
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
) -> Option<Vec2> {
    obb_obb_mtv(
        center,
        angle,
        half,
        Vec2::new(rx + 0.5 * rw, ry + 0.5 * rh),
        0.0,
        (0.5 * rw, 0.5 * rh),
    )
}

/// Normalize angle to [-PI, PI]
#[inline]
pub fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= 2.0 * std::f32::consts::PI;
    }
    while angle < -std::f32::consts::PI {
        angle += 2.0 * std::f32::consts::PI;
    }
    angle
}

/// Clamps value strictly to [-1.0, 1.0]
#[inline]
pub fn clamp_action(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(-1.0, 1.0)
    }
}
