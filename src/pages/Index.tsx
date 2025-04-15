
import React from 'react';
import Navbar from '@/components/Navbar';
import FloatingSpheres from '@/components/FloatingSpheres';
import Hero from '@/components/Hero';
import Features from '@/components/Features';
import Footer from '@/components/Footer';

const Index = () => {
  return (
    <div className="bg-tictac-dark min-h-screen overflow-hidden relative">
      {/* Dynamic gradient background */}
      <div className="fixed inset-0 z-0 gradient-bg opacity-30"></div>
      
      {/* Moving spheres in the background */}
      <FloatingSpheres />
      
      {/* Main content */}
      <div className="relative z-10">
        <Navbar />
        
        <main>
          <Hero />
          <Features />
        </main>
        
        <Footer />
      </div>
    </div>
  );
};

export default Index;
