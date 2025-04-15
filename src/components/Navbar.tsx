
import React from 'react';
import { Button } from "@/components/ui/button";
import { Zap } from "lucide-react";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 py-4 px-6 md:px-12 glass-effect">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Zap className="w-6 h-6 text-tictac-purple" />
          <span className="text-xl font-bold tracking-tighter">ULTIMATE TIC-TAC-TOE</span>
        </div>
        
        <div className="hidden md:flex items-center space-x-8">
          <a href="#" className="nav-link font-medium text-white hover:text-tictac-purple">Learn</a>
          <a href="#" className="nav-link font-medium text-white hover:text-tictac-purple">Play</a>
          <a href="#" className="nav-link font-medium text-white hover:text-tictac-purple">Explore</a>
        </div>
        
        <div>
          <Button className="bg-tictac-purple hover:bg-tictac-purple-dark text-white animated-button">
            Play Now
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
