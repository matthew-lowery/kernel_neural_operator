import jax
from jax import numpy as jnp
DTYPE = jnp.float64 ### 4 quad rule
jax.config.update("jax_enable_x64", True)

def triangle_quad_rule(
    n, quadrature_fn, triangle=jnp.array([[0, 0], [1, 0], [0.5, jnp.sqrt(3) / 2]])
):
    """
    Defines a quadrature rule for the reference triangle ([0,0],[1,0],[0.5,sqrt(3)/2]])
    by creating rules for 3 quadrilaterals within this triangle.
    """

    ### tensor-product approach to defining the rule for the unit-square
    def quad_rule_2d(n, quadrature_fn):
        ndims = 2
        quad_nodes, quad_weights = quadrature_fn(n)
        a, b = -1, 1  ### old domain
        c, d = 0, 1  ### new domain
        t, w = quad_nodes, quad_weights
        t = (((t - a) * (d - c)) / (b - a)) + c
        det_j = (d - c) / (b - a)
        w *= det_j
        t = jnp.array(jnp.meshgrid(*[t] * ndims))
        t = t.reshape(len(t), -1).T
        w = jnp.outer(*([w] * ndims)).flatten()[:, None]
        quad_rule_2d = (t, w)
        return quad_rule_2d

    quad_rule = quad_rule_2d(n, quadrature_fn)

    ### map a single coordinate in the unit square to an arbitrary quadrilateral (defined by 4 vertices)
    def coord_square_to_quadrilateral(x, quadrilateral):
        x1, x2, x3, x4 = quadrilateral
        xi, eta = x
        psi_1 = lambda xi, eta: (1 - xi) * (1 - eta)
        psi_2 = lambda xi, eta: xi * (1 - eta)
        psi_3 = lambda xi, eta: xi * eta
        psi_4 = lambda xi, eta: (1 - xi) * eta
        return (
            x1 * psi_1(xi, eta)
            + x2 * psi_2(xi, eta)
            + x3 * psi_3(xi, eta)
            + x4 * psi_4(xi, eta)
        )

    ### jacobian determinant of the transformation of a unit square to an arbitrary quadrilateral
    def detj_square_to_quadrilateral(x, quadrilateral):
        J = jax.jacfwd(coord_square_to_quadrilateral, argnums=0)(x, quadrilateral)
        return jnp.linalg.det(J)

    ### using the two functions above to map the unit square quad rule to a quad rule for an arbitrary quadrilat
    def quad_rule_square_to_quadrilateral(quad_rule, quadrilateral):
        t, w = quad_rule
        updated_w = jax.vmap(
            lambda w, t, quadrilateral: w
            * detj_square_to_quadrilateral(t, quadrilateral),
            in_axes=[0, 0, None],
        )(w, t, quadrilateral)
        updated_t = jax.vmap(coord_square_to_quadrilateral, in_axes=[0, None])(
            t, quadrilateral
        )
        return (updated_t, updated_w)

    ### points for an equilateral triangle
    A, B, C = triangle
    O = (A + B + C) / 3
    D = (A + B) / 2
    E = (B + C) / 2
    F = (A + C) / 2
    ### 3 quadrilaterals within the reference triangle
    quadrilaterals = jnp.array([[A, D, O, F], [B, E, O, D], [C, F, O, E]])

    ### map the unit square quad rule to each of these quadrilaterals to make a rule for the reference triangle
    triangle_quad_t, triangle_quad_w = jax.vmap(
        quad_rule_square_to_quadrilateral, in_axes=[None, 0]
    )(quad_rule, quadrilaterals)
    triangle_quad_rule = (
        triangle_quad_t.reshape(-1, 2).astype(DTYPE),
        triangle_quad_w.flatten()[:, None].astype(DTYPE),
    )
    return triangle_quad_rule