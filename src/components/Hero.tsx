
import React from 'react';
import { Button } from "@/components/ui/button";
import { ArrowRight, Zap } from "lucide-react";

const Hero = () => {
  return (
    <div className="relative z-10 min-h-screen flex flex-col items-center justify-center text-center px-4">
      <div className="space-y-2">
        <p className="text-tictac-purple font-medium uppercase tracking-wider">WELCOME TO ULTIMATE TIC-TAC-TOE</p>
        
        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter mt-4 mb-6 text-white">
          Strategic<br/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-tictac-purple to-tictac-blue-light">
            Tic-Tac-Toe.
          </span>
        </h1>
        
        <p className="text-lg md:text-xl max-w-2xl mx-auto text-white/80 mb-8">
          A next-level experience of the classic game with advanced strategy,
          multiple boards, and competitive gameplay.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button className="bg-tictac-purple hover:bg-tictac-purple-dark text-white px-8 py-6 text-lg animated-button">
            Play Now <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <Button variant="outline" className="bg-transparent border-white/20 hover:bg-white/10 text-white px-8 py-6 text-lg">
            Learn Rules
          </Button>
        </div>
      </div>
      
      <div className="mt-16 glass-effect p-4 rounded-lg max-w-md">
        <div className="flex items-center mb-2">
          <Zap className="h-5 w-5 text-tictac-purple mr-2" />
          <p className="font-medium">Quick Stats</p>
        </div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-tictac-purple text-2xl font-bold">10K+</p>
            <p className="text-xs text-white/70">Daily Games</p>
          </div>
          <div>
            <p className="text-tictac-purple text-2xl font-bold">24/7</p>
            <p className="text-xs text-white/70">Available</p>
          </div>
          <div>
            <p className="text-tictac-purple text-2xl font-bold">1.2M</p>
            <p className="text-xs text-white/70">Players</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
