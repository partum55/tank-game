use tank_core::math::{Vec2, ray_circle_intersect};

#[test]
fn test_vec2_operations() {
    let a = Vec2::new(3.0, 4.0);
    assert!((a.length() - 5.0).abs() < 1e-5);
    let n = a.normalize();
    assert!((n.length() - 1.0).abs() < 1e-5);
}

#[test]
fn test_ray_circle_intersection() {
    let ray_origin = Vec2::new(0.0, 0.0);
    let ray_dir = Vec2::new(1.0, 0.0);
    let circle_center = Vec2::new(10.0, 0.0);
    let radius = 2.0;
    let hit = ray_circle_intersect(ray_origin, ray_dir, circle_center, radius);
    assert!(hit.is_some());
    let (t, _hit_point) = hit.unwrap();
    assert!((t - 8.0).abs() < 1e-4);
}
