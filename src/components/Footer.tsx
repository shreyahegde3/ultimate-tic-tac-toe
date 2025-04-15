
import React from 'react';
import { Zap, Github, Twitter, Instagram } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="relative z-10 px-6 py-12 border-t border-white/10">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <Zap className="w-5 h-5 text-tictac-purple" />
              <span className="font-bold tracking-tighter">ULTIMATE TIC-TAC-TOE</span>
            </div>
            <p className="text-white/70 text-sm">
              A next-level experience of the classic game with advanced strategy and competitive gameplay.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium mb-4">Play</h3>
            <ul className="space-y-2 text-white/70">
              <li><a href="#" className="hover:text-tictac-purple transition-colors">Single Player</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors">Multiplayer</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors">Tournaments</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-medium mb-4">Learn</h3>
            <ul className="space-y-2 text-white/70">
              <li><a href="#" className="hover:text-tictac-purple transition-colors">How to Play</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors">Strategy Guide</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors">FAQ</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-medium mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a href="#" className="text-white/70 hover:text-tictac-purple transition-colors">
                <Github className="h-5 w-5" />
              </a>
              <a href="#" className="text-white/70 hover:text-tictac-purple transition-colors">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-white/70 hover:text-tictac-purple transition-colors">
                <Instagram className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>
        
        <div className="mt-12 pt-6 border-t border-white/10 text-center text-white/50 text-sm">
          <p>© {new Date().getFullYear()} Ultimate Tic-Tac-Toe. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
