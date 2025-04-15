
import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Zap, Menu, X } from "lucide-react";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [scrolled]);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 py-6 px-6 md:px-12 transition-all duration-300 ${
      scrolled ? 'glass-effect backdrop-blur-xl' : 'bg-transparent'
    }`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Zap className="w-6 h-6 text-tictac-purple animate-pulse-slow" />
          <span className="text-xl font-bold tracking-tighter">ULTIMATE TIC-TAC-TOE</span>
        </div>
        
        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-16">
          <a href="#learn" className="nav-link font-medium text-white hover:text-tictac-purple transition-colors duration-300">Learn</a>
          <a href="#play" className="nav-link font-medium text-white hover:text-tictac-purple transition-colors duration-300">Play</a>
          <a href="#explore" className="nav-link font-medium text-white hover:text-tictac-purple transition-colors duration-300">Explore</a>
        </div>
        
        {/* Mobile Menu Button */}
        <div className="md:hidden">
          <Button 
            variant="ghost"
            size="icon"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="text-white"
          >
            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </Button>
        </div>
        
        {/* Play Now Button (desktop) */}
        <div className="hidden md:block">
          <Button className="bg-gradient-to-r from-tictac-purple to-tictac-purple-dark text-white animated-button">
            Play Now
          </Button>
        </div>
      </div>
      
      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden glass-effect mt-4 p-4 rounded-lg animate-fade-in">
          <div className="flex flex-col space-y-4">
            <a href="#learn" className="font-medium text-white py-2 px-4 hover:bg-white/10 rounded transition-colors" onClick={() => setMobileMenuOpen(false)}>Learn</a>
            <a href="#play" className="font-medium text-white py-2 px-4 hover:bg-white/10 rounded transition-colors" onClick={() => setMobileMenuOpen(false)}>Play</a>
            <a href="#explore" className="font-medium text-white py-2 px-4 hover:bg-white/10 rounded transition-colors" onClick={() => setMobileMenuOpen(false)}>Explore</a>
            <Button className="bg-tictac-purple hover:bg-tictac-purple-dark text-white w-full animated-button">
              Play Now
            </Button>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
