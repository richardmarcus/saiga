/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{


#define IM_PI                           3.14159265358979323846f
// Forward declaration of SphericalParameters
template <typename T>
struct SphericalParameters;
// Basic Intrinsics Matrix K used in the pinhole camera model.
//
// |  fx   s   cx  |
// |   0   fy  cy  |
// |   0   0    1  |
//
// The inverse is:
// |   1/fx   -s/(fx*fy)  (s*cy-cx*fy)/(fx*fy)  |
// |   0      1/fy     -cy/fy                |
// |   0      0        1                     |
template <typename T>
struct IntrinsicsPinhole
{
    using Vec5 = Eigen::Matrix<T, 5, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;
    using Mat2 = Eigen::Matrix<T, 2, 2>;

    // Initialized to identity
    T fx = 1, fy = 1;
    T cx = 0, cy = 0;
    T s = 0;

    HD inline IntrinsicsPinhole() {}
    HD inline IntrinsicsPinhole(T fx, T fy, T cx, T cy, T s) : fx(fx), fy(fy), cx(cx), cy(cy), s(s) {}
    HD inline IntrinsicsPinhole(const Vec5& v) : fx(v(0)), fy(v(1)), cx(v(2)), cy(v(3)), s(v(4)) {}
    HD inline IntrinsicsPinhole(const Mat3& K) : fx(K(0, 0)), fy(K(1, 1)), cx(K(0, 2)), cy(K(1, 2)), s(K(0, 1)) {}


    /*
    // Conversion method to SphericalParameters
    HD inline SphericalParameters<T> toSpherical() const {
        T fx_spherical = this->fx;  
        T fy_spherical = this->fy;
        T cx_spherical = this->cx;  // Copy the principal point offsets directly
        T cy_spherical = this->cy;
        T s_spherical  = this->s;   // Use the skew directly

        return SphericalParameters<T>(fx_spherical, fy_spherical, cx_spherical, cy_spherical, s_spherical);
    }*/

    HD inline IntrinsicsPinhole<T> inverse() const
    {
        T fx2 = 1 / fx;
        T fy2 = 1 / fy;
        T cx2 = (s * cy - cx * fy) / (fx * fy);
        T cy2 = -cy / fy;
        T s2  = -s / (fx * fy);
        return {fx2, fy2, cx2, cy2, s2};
    }

    HD inline Vec2 project(const Vec3& X) const
    {
        auto invz = T(1) / X(2);
        auto x    = X(0) * invz;
        auto y    = X(1) * invz;
        return {fx * x + s * y + cx, fy * y + cy};
    }

    HD inline Vec2 operator*(const Vec3& X) const { return project(X); }

    // same as above, but keep the z value in the output
    HD inline Vec3 project3(const Vec3& X) const
    {
        Vec2 p = project(X);
        return {p(0), p(1), X(2)};
    }

    HD inline Vec3 unproject(const Vec2& ip, T depth) const
    {
        Vec3 p((ip(0) - cx) / fx, (ip(1) - cy) / fy, 1);
        return p * depth;
    }

    // from image to normalized image space
    // same as above but without depth
    HD inline Vec2 unproject2(const Vec2& ip) const { return {(ip(0) - cx) / fx, (ip(1) - cy) / fy}; }

    HD inline Vec2 normalizedToImage(const Vec2& p) const { return {fx * p(0) + s * p(1) + cx, fy * p(1) + cy}; }


    [[nodiscard]] HD inline IntrinsicsPinhole<T> scale(T s) const { return IntrinsicsPinhole<T>(Vec5(coeffs() * s)); }
    


    HD inline Vec2 normalizedToImage(const Vec2& p, Mat2* J_point, Matrix<T, 2, 5>* J_K) const
    {
        const Vec2 image_point = normalizedToImage(p);

        if (J_point)
        {
            (*J_point)(0, 0) = fx;
            (*J_point)(0, 1) = s;
            (*J_point)(1, 0) = 0;
            (*J_point)(1, 1) = fy;
        }

        if (J_K)
        {
            (*J_K)(0, 0) = p(0);
            (*J_K)(0, 1) = 0;
            (*J_K)(0, 2) = 1;
            (*J_K)(0, 3) = 0;
            (*J_K)(0, 4) = p(1);

            (*J_K)(1, 0) = 0;
            (*J_K)(1, 1) = p(1);
            (*J_K)(1, 2) = 0;
            (*J_K)(1, 3) = 1;
            (*J_K)(1, 4) = 0;
        }

        return image_point;
    }




    HD inline Mat3 matrix() const
    {
        Mat3 k;

        //   fx, s,  cx,
        //     0,  fy, cy,
        //     0,  0,  1;
        k(0, 0) = fx;
        k(0, 1) = s;
        k(0, 2) = cx;

        k(1, 0) = 0;
        k(1, 1) = fy;
        k(1, 2) = cy;

        k(2, 0) = 0;
        k(2, 1) = 0;
        k(2, 2) = 1;
        return k;
    }

    template <typename G>
    HD inline IntrinsicsPinhole<G> cast() const
    {
        return {(G)fx, (G)fy, (G)cx, (G)cy, (G)s};
    }

    // convert to eigen vector
    HD inline Vec5 coeffs() const { return {fx, fy, cx, cy, s}; }
    HD inline void coeffs(const Vec5& v) { (*this) = v; }

    bool operator==(const IntrinsicsPinhole& other) const
    {
        bool is_equal = true;
        is_equal &= (fx == other.fx);
        is_equal &= (fy == other.fy);
        is_equal &= (cx == other.cx);
        is_equal &= (cy == other.cy);
        is_equal &= (s == other.s);
        return is_equal;
    }
    bool operator!=(const IntrinsicsPinhole& other) const { return !(other == *this); }
};

template <typename T>
HD inline IntrinsicsPinhole<T> operator*(const IntrinsicsPinhole<T>& l, const IntrinsicsPinhole<T>& r)
{
    T fx2 = l.fx * r.fx;
    T fy2 = l.fy * r.fy;
    T s2  = l.fx * r.s + l.s * r.fy;
    T cx2 = l.fx * r.cx + l.s * r.fy + l.cx;
    T cy2 = l.fy * r.cy + l.cy;
    return {fx2, fy2, cx2, cy2, s2};
}

using IntrinsicsPinholed = IntrinsicsPinhole<double>;
using IntrinsicsPinholef = IntrinsicsPinhole<float>;



template <typename T>
std::ostream& operator<<(std::ostream& strm, const IntrinsicsPinhole<T> intr)
{
    strm << intr.coeffs().transpose();
    return strm;
}


template <typename T>
struct StereoCamera4Base : public IntrinsicsPinhole<T>
{
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Base = IntrinsicsPinhole<T>;
    using Base::cx;
    using Base::cy;
    using Base::fx;
    using Base::fy;
    using Base::s;
    using Vec3 = typename Base::Vec3;
    /**
     * Baseline * FocalLength_x
     *
     * Used to convert from depth to disparity:
     * disparity = bf / depth
     * depth = bf / disparity
     */
    T bf;
    StereoCamera4Base() {}
    StereoCamera4Base(T fx, T fy, T cx, T cy, T s, T bf) : IntrinsicsPinhole<T>(fx, fy, cx, cy, s), bf(bf) {}
    StereoCamera4Base(const IntrinsicsPinhole<T>& i, T bf) : IntrinsicsPinhole<T>(i), bf(bf) {}
    StereoCamera4Base(const Vec6& v) : IntrinsicsPinhole<T>(Vec5(v.template head<5>())), bf(v(5)) {}

    StereoCamera4Base<T> inverse() const
    {
        T fx2 = 1 / fx;
        T fy2 = 1 / fy;
        T cx2 = (s * cy - cx * fy) / (fx * fy);
        T cy2 = -cy / fy;
        T s2  = -s / (fx * fy);
        T bf2 = 1 / bf;
        return {fx2, fy2, cx2, cy2, s2, bf2};
    }

    template <typename G>
    StereoCamera4Base<G> cast()
    {
        return {IntrinsicsPinhole<T>::template cast<G>(), static_cast<G>(bf)};
    }

    // convert an image point + its depth to the predicted point in the right image.
    // We subtract the disparity
    double LeftPointToRight(double x, double z) const { return x - bf / z; }

    // depth     = baseline * fx / disparity
    // disparity = baseline * fx / depth
    double DepthToDisparity(double depth) { return bf / depth; }
    double DisparityToDepth(double disparity) { return bf / disparity; }

    Vec3 projectStereo(const Vec3& X) const
    {
        auto invz        = T(1) / X(2);
        auto x           = X(0) * invz;
        auto y           = X(1) * invz;
        x                = fx * x + cx;
        y                = fy * y + cy;
        auto stereoPoint = x - bf * invz;
        return {x, y, stereoPoint};
    }

    Vec6 coeffs() const
    {
        Vec6 v;
        v << this->fx, this->fy, this->cx, this->cy, this->s, bf;
        return v;
    }
    void coeffs(const Vec6& v) { (*this) = v; }

    // Baseline in meters
    T baseLine() { return bf / fx; }
};

using StereoCamera4  = StereoCamera4Base<double>;
using StereoCamera4f = StereoCamera4Base<float>;

template <typename T>
std::ostream& operator<<(std::ostream& strm, const StereoCamera4Base<T> intr)
{
    strm << intr.coeffs().transpose();
    return strm;
}

// intrinsics are used to convert between pixel coordinates and world coordinates
// inverse intrinsics do world to pixel
template <typename T>
struct SphericalParameters
{
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    // f is lidar fov, c is offset, both in rad/pi, s = center_offset
    T fx = 2, fy = 1;
    T cx = 0, cy = 0;
    T w = 1024, h = 64;

    HD inline SphericalParameters() {}
    HD inline SphericalParameters(T fx, T fy, T cx, T cy, T w, T h) : fx(fx), fy(fy), cx(cx), cy(cy), w(w), h(h) {}
    HD inline SphericalParameters(const Vec6 &v) : fx(v(0)), fy(v(1)), cx(v(2)), cy(v(3)), w(v(4)), h(v(5)) {}


    HD inline Vec6 coeffs() const { return {fx, fy, cx, cy, w, h}; }
    HD inline void coeffs(const Vec6& v) { (*this) = v; }
    template <typename G>
    HD inline SphericalParameters<G> cast() const
    {
        return {(G)fx, (G)fy, (G)cx, (G)cy, (G)w, (G)h};
    }

    //works if we just want to apply zoom and offset, just leave out s
    HD inline Vec2 normalizedToImage(const Vec2& p) const { return {fx * p(0)+ cx, fy * p(1) + cy}; }

    HD inline Vec2 normalizedToImage(const Vec2& p, Mat2* J_point, Matrix<T, 2, 5>* J_K) const
    {
        const Vec2 image_point = normalizedToImage(p);

        if (J_point)
        {
            (*J_point)(0, 0) = fx;
            (*J_point)(0, 1) = 0;
            (*J_point)(1, 0) = 0;
            (*J_point)(1, 1) = fy;
        }

        if (J_K)
        {

        }

        return image_point;
    }
    HD inline Vec2 toPano(const Vec3& p) const
    {
        float azimuth = atan2f(p(0), p(2)) / (fx * M_PI) + cx + 0.5;
        float elevation = asinf(normalize(p)(1)) / (fy * M_PI) + cy;

        // Multiply azimuth by w and elevation by h
        azimuth *= w;
        elevation *= h;

        return {azimuth, elevation};
    }

    HD inline Vec2 toPano(const Vec3& p, Matrix<T, 2, 3>* J_spherical) const
    {
        const Vec2 image_point = toPano(p);

        T norm_p = p.norm();

        // Compute azimuth partial derivatives
        T atan2_der_p0 = p(0) / (p(2) * p(2) + p(0) * p(0));
        T atan2_der_p2 = p(2) / (p(2) * p(2) + p(0) * p(0));

        // Multiply azimuth derivative by w
        (*J_spherical)(0, 0) = (atan2_der_p0 / (fx * M_PI)) * w;  // Partial derivative of azimuth w.r.t. p(2)
        (*J_spherical)(0, 1) = 0;                                 // Partial derivative of azimuth w.r.t. p(1)
        (*J_spherical)(0, 2) = (atan2_der_p2 / (fx * M_PI)) * w;  // Partial derivative of azimuth w.r.t. p(0)

        // Compute elevation partial derivatives
        T sin_term = p(1) / norm_p;
        T sqrt_term = std::sqrt(1 - sin_term * sin_term);

        T asinf_der_p0 = p(2) * p(1) / (norm_p * norm_p * norm_p * sqrt_term);
        T asinf_der_p1 = (norm_p * norm_p - p(1) * p(1)) / (norm_p * norm_p * norm_p * sqrt_term);
        T asinf_der_p2 = p(1) * p(0) / (norm_p * norm_p * norm_p * sqrt_term);

        // Multiply elevation derivative by h
        (*J_spherical)(1, 0) = (asinf_der_p0 / (fy * M_PI)) * h;  // Partial derivative of elevation w.r.t. p(2)
        (*J_spherical)(1, 1) = (asinf_der_p1 / (fy * M_PI)) * h;  // Partial derivative of elevation w.r.t. p(1)
        (*J_spherical)(1, 2) = (asinf_der_p2 / (fy * M_PI)) * h;  // Partial derivative of elevation w.r.t. p(0)

        return image_point;
    }

};




using SphericalParametersd = SphericalParameters<double>;
using SphericalParametersf = SphericalParameters<float>;

template <typename T>
std::ostream& operator<<(std::ostream& strm, const SphericalParameters<T> intr)
{
    strm << intr.coeffs().transpose();
    return strm;
}

}  // namespace Saiga
