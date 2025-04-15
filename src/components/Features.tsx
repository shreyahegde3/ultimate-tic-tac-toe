
import React from 'react';
import { 
  Trophy, 
  Zap, 
  Users, 
  Brain, 
  Layers 
} from "lucide-react";

interface FeatureProps {
  title: string;
  description: string;
  icon: React.ReactNode;
}

const Feature: React.FC<FeatureProps> = ({ title, description, icon }) => {
  return (
    <div className="glass-effect p-6 rounded-lg hover:border-tictac-purple/30 transition-all duration-300">
      <div className="h-12 w-12 rounded-full bg-tictac-purple/20 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-white/70">{description}</p>
    </div>
  );
};

const Features = () => {
  return (
    <div className="relative z-10 py-20 px-4">
      <div className="text-center mb-16">
        <h2 className="text-3xl md:text-4xl font-bold mb-4">Game Features</h2>
        <p className="text-white/70 max-w-xl mx-auto">
          Ultimate Tic-Tac-Toe reimagines the classic game with strategic depth
          and competitive elements for the modern player
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <Feature 
          title="Strategic Depth" 
          description="Play on nine boards simultaneously with nested gameplay that requires advanced planning and tactics."
          icon={<Brain className="h-6 w-6 text-tictac-purple" />}
        />
        
        <Feature 
          title="Competitive Rankings" 
          description="Climb the global leaderboards and earn achievements as you master the game."
          icon={<Trophy className="h-6 w-6 text-tictac-purple" />}
        />
        
        <Feature 
          title="Multiplayer" 
          description="Challenge friends or play against opponents of your skill level from around the world."
          icon={<Users className="h-6 w-6 text-tictac-purple" />}
        />
        
        <Feature 
          title="Speed Modes" 
          description="Test your skills in timed matches that require quick thinking and faster decisions."
          icon={<Zap className="h-6 w-6 text-tictac-purple" />}
        />
        
        <Feature 
          title="Customizable Experience" 
          description="Choose from various themes, board layouts, and gameplay modes to personalize your experience."
          icon={<Layers className="h-6 w-6 text-tictac-purple" />}
        />
      </div>
    </div>
  );
};

export default Features;
