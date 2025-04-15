
import React from 'react';

interface SphereProps {
  size: string;
  color: string;
  top: string;
  left: string;
  delay?: string;
  duration?: string;
  opacity?: string;
}

const Sphere: React.FC<SphereProps> = ({ size, color, top, left, delay = '0s', duration = '6s', opacity = '0.7' }) => {
  return (
    <div 
      className="absolute rounded-full floating-sphere animate-float"
      style={{
        width: size,
        height: size,
        backgroundColor: color,
        top,
        left,
        opacity,
        animationDelay: delay,
        animationDuration: duration,
      }}
    />
  );
};

const FloatingSpheres: React.FC = () => {
  return (
    <div className="fixed inset-0 overflow-hidden z-0">
      <Sphere size="120px" color="#9b87f5" top="15%" left="10%" delay="0s" opacity="0.5" />
      <Sphere size="80px" color="#0EA5E9" top="70%" left="20%" delay="1s" opacity="0.6" />
      <Sphere size="150px" color="#7E69AB" top="40%" left="80%" delay="2s" opacity="0.4" />
      <Sphere size="100px" color="#F97316" top="80%" left="85%" delay="0.5s" opacity="0.5" />
      <Sphere size="60px" color="#33C3F0" top="25%" left="50%" delay="1.5s" opacity="0.7" />
      <Sphere size="90px" color="#8B5CF6" top="60%" left="40%" delay="3s" opacity="0.5" />
      <Sphere size="70px" color="#F97316" top="10%" left="70%" delay="2s" opacity="0.6" />
    </div>
  );
};

export default FloatingSpheres;
