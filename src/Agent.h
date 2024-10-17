/*
 * Agent.h
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#ifndef RVO_AGENT_H_
#define RVO_AGENT_H_

/**
 * \file       Agent.h
 * \brief      Contains the Agent class.
 */

#include "Definitions.h"
#include "RVOSimulator.h"

static const float EPSILON = 0.001f;
namespace RVO {
	/**
	 * \brief      Defines an agent in the simulation.
	 */
	class Agent {
	private:
		/**
		 * \brief      Constructs an agent instance.
		 * \param      sim             The simulator instance.
		 */
		explicit Agent(RVOSimulator *sim);

		/**
		 * \brief      Computes the neighbors of this agent.
		 */
		void computeNeighbors();

		/**
		 * \brief      Computes the new velocity of this agent.
		 */
		void computeNewVelocity();

		/**
		 * \brief      Inserts an agent neighbor into the set of neighbors of
		 *             this agent.
		 * \param      agent           A pointer to the agent to be inserted.
		 * \param      rangeSq         The squared range around this agent.
		 */
		void insertAgentNeighbor(const Agent *agent, float &rangeSq);

		/**
		 * \brief      Inserts a static obstacle neighbor into the set of neighbors
		 *             of this agent.
		 * \param      obstacle        The number of the static obstacle to be
		 *                             inserted.
		 * \param      rangeSq         The squared range around this agent.
		 */
		void insertObstacleNeighbor(const Obstacle *obstacle, float rangeSq);

		/**
		 * \brief      Updates the two-dimensional position and two-dimensional
		 *             velocity of this agent.
		 */
		void update();

		/**
		 * \brief      Adds the constrains for non-holomonic agents based on the nearest agent/obst
		 * \param      min_dist        The minimum distance to keep from the nearest agent/obst.
		 *
		 */
		void addNHConstraints(double min_dist);
		
		/**
		 * \brief      Non-Holonomic対応用のパラメーターを初期化
		 * \param      minErrorHolo 	実際は衝突する際に使われるため、つまりぶつかってから離れる距離
		 * \param      maxErrorHolo 	論文中のエプシロン、Holoとの最大誤差距離
		 * \param      velMaxW			論文中のVmax,w
		 * \param      wMax				論文中のWmax、最大角速度、2Vmax/lw、lwは車輪の幅
		 * \param      curAllowedError	現在の誤差許容範囲？不明
		 * \param      timeToHolo		論文中のT、Holonomicの向きになるまでの時間
		 *
		 */
		void setHoloParams(double minErrorHolo, double maxErrorHolo, double velMaxW, double wMax, double curAllowedError, double timeToHolo);

		void setAgentAngularInfo(double heading, double angVel);

		std::vector<std::pair<float, const Agent *> > agentNeighbors_;
		size_t maxNeighbors_;
		float maxSpeed_;
		float neighborDist_;
		Vector2 newVelocity_;
		std::vector<std::pair<float, const Obstacle *> > obstacleNeighbors_;
		std::vector<Line> orcaLines_;
		std::vector<Line> additionalOrcaLines_; // 追加したOrcaLines
		Vector2 position_;
		Vector2 prefVelocity_;
		float radius_;
		RVOSimulator *sim_;
		float timeHorizon_;
		float timeHorizonObst_;
		Vector2 velocity_;
		float collabCoeff_;
		bool isDeadLock_; // DeadLock処理用フラグ
		double minErrorHolo_; // Holonomic用の最小誤差
		double maxErrorHolo_;
		double velMaxW_; // 論文のv_max_w
		double wMax_; // 最大回転速度
		double curAllowedError_;
		double timeToHolo_; // Holonomicの向きになるまでの時間　T
		float heading_; // Agentの向き
		float angVel_; // Agentの今の角速度

		size_t id_;

		friend class KdTree;
		friend class RVOSimulator;
	};

	/**
	 * \relates    Agent
	 * \brief      Solves a one-dimensional linear program on a specified line
	 *             subject to linear constraints defined by lines and a circular
	 *             constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      lineNo        The specified line constraint.
	 * \param      radius        The radius of the circular constraint.
	 * \param      optVelocity   The optimization velocity.
	 * \param      directionOpt  True if the direction should be optimized.
	 * \param      result        A reference to the result of the linear program.
	 * \return     True if successful.
	 */
	bool linearProgram1(const std::vector<Line> &lines, size_t lineNo,
						float radius, const Vector2 &optVelocity,
						bool directionOpt, Vector2 &result);

	/**
	 * \relates    Agent
	 * \brief      Solves a two-dimensional linear program subject to linear
	 *             constraints defined by lines and a circular constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      radius        The radius of the circular constraint.
	 * \param      optVelocity   The optimization velocity.
	 * \param      directionOpt  True if the direction should be optimized.
	 * \param      result        A reference to the result of the linear program.
	 * \return     The number of the line it fails on, and the number of lines if successful.
	 */
	size_t linearProgram2(const std::vector<Line> &lines, float radius,
						  const Vector2 &optVelocity, bool directionOpt,
						  Vector2 &result);

	/**
	 * \relates    Agent
	 * \brief      Solves a two-dimensional linear program subject to linear
	 *             constraints defined by lines and a circular constraint.
	 * \param      lines         Lines defining the linear constraints.
	 * \param      numObstLines  Count of obstacle lines.
	 * \param      beginLine     The line on which the 2-d linear program failed.
	 * \param      radius        The radius of the circular constraint.
	 * \param      result        A reference to the result of the linear program.
	 */
	void linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine,
						float radius, Vector2 &result);
}

// NH計算用の関数
void addMovementConstraintsDiffSimple(double max_track_speed, double heading, std::vector<RVO::Line>& additional_orca_lines);
void addMovementConstraintsDiff(double error, double T, double max_vel_x, double max_vel_th, double heading, double v_max_ang, std::vector<RVO::Line>& additional_orca_lines);
double beta(double T, double theta, double v_max_ang);
double gamma(double T, double theta, double error, double v_max_ang);
double calcVstar(double vh, double theta);
double calcVstarError(double T,double theta, double error);
double calculateMaxTrackSpeedAngle(double T, double theta, double error, double max_vel_x, double max_vel_th, double v_max_ang);
double sign(double x);

#endif /* RVO_AGENT_H_ */
