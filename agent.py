import torch.nn as nn
import torch
from holdem import *
import multiprocessing

NUM_PROCESSES = 4

pathp = 'D:\DRL\Hold\'em\Policy.pt'
pathv = 'D:\DRL\Hold\'em\Value.pt'
pathg = 'D:\DRL\Hold\'em\Guess.pt'


class CardGuesser(nn.Module):
    def __init__(self):
        super(CardGuesser, self).__init__()
        self.card1 = nn.Sequential(
            nn.Linear(54, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 52)
        )

    def forward(self, x):
        return self.card1(x)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(82, 256),  # one-hot(52) + 9*players(activity, fold_or_not, total_bet/original) + (pot/origin=reward) + (call/origin=risk)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # call, check, fold, 61*stack if raise
        )

    def forward(self, x):
        return self.policy(x)


class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(82, 256),  # action(64) + one-hot(52) + 9*players(activity, fold_or_not, total_bet/original) + (pot/origin=reward) + (call/origin=risk)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.value(x)


class Agent:
    def __init__(self, policy, value, guesser):
        self.p = policy
        self.g = guesser
        self.v = value


    def policy(self, id, current_game):
        total_info = torch.zeros(size=(9, 3))
        for i in range(current_game.num_players-1):
            idx = (i+id+1) % current_game.num_players
            total_info[i, 0] = 1
            total_info[i, 1] = current_game.unfold_list[idx]
            total_info[i, 2] = 1 - current_game.players_pocket[idx] / current_game.players_origin[idx]
        total_info = total_info[torch.randperm(total_info.size()[0])]
        input = torch.cat(
            (torch.tensor([current_game.players[id].hand_comm]),
             total_info.reshape((1, 27)),
             torch.tensor([[max(1.01, current_game.call / (current_game.players_pocket[id]+0.01))]]),
             torch.tensor(([[1 - current_game.players_pocket[id] / current_game.players_origin[id]]])),
             torch.tensor([[(sum(current_game.chips) + current_game.pot) / current_game.players_origin[id]]])),
            1)
        out = self.p.forward(input)
        exp_out = torch.exp(out)

        return exp_out, input

    def guesser(self, input):
        return torch.exp(self.g.forward(input))

    def value(self, input):
        # act = torch.zeros(size=(1, 64))
        # act[0, action] = 1
        # putin = torch.cat((act, input), 1)
        return self.v.forward(input)

    def mcts(self,phase, agent, guessrecord, id, current_game, st):
        return monte_carlo(phase, agent, guessrecord, id, current_game, search_time=st)


def value_sampler(id, exp_out, current_game):
    if current_game.players_pocket[id] < current_game.call:
        for t in range(2, 64):
            exp_out[0, t] = 0
    prob = exp_out / torch.sum(exp_out)
    # print(prob)
    temp = torch.multinomial(prob, 1).item()
    if temp == 0:
        bet = ('fold', 0)
    elif temp == 1:
        bet = ('check', 0)
    elif temp == 2:
        bet = ('call', current_game.call)
    else:
        bet = (
        'raise', int((temp - 2) * (current_game.players_pocket[id] - current_game.call) / 100) + current_game.call)
    return bet, temp


def mcts_sample(id, points, current_game):
    if current_game.players_pocket[id] < current_game.call:
        for t in range(2, 64):
            points[t] = float('-inf')
    # print(points)
    temp = points.index(max(points))
    points[temp] = float('-inf')
    if temp == 0:
        bet = ('fold', 0)
    elif temp == 1:
        bet = ('check', 0)
    elif temp == 2:
        bet = ('call', current_game.call)
    else:
        bet = (
        'raise', int((temp - 2) * (current_game.players_pocket[id] - current_game.call) / 100) + current_game.call)
    return bet, temp, points


def player_copier(players: tuple) -> tuple:
    res = ()
    for i in range(len(players)):
        copy = Player(players[i].money)
        copy.hand = [0 for _ in range(52)]
        copy.hand_comm = [0 for _ in range(52)]
        copy.hc = None
        copy.hc_show = None
        res += (copy,)
    return res


def game_copier(id, current_game: Game, players: tuple) -> Game:
    deck = Deck()
    deck.deck = players[id].hand_comm[:]
    # print(players[id].hand_comm)
    deck.comm = current_game.community[:]
    copy = Game(players, deck)
    copy.button = current_game.button
    copy.round = current_game.round
    copy.chips = current_game.chips[:]
    copy.unfold_list = current_game.unfold_list[:]
    copy.community = current_game.community[:]
    copy.comm_show = current_game.comm_show[:]
    copy.pot = current_game.pot
    copy.minbet = current_game.minbet
    copy.call = current_game.call
    copy.minraise = current_game.minraise
    copy.players_pocket = current_game.players_pocket[:]
    copy.players_origin = current_game.players_origin[:]

    return copy


def sim(phase, agent, id, game_copy, ratio=4, multi=0.02):
    button = game_copy.button
    idx = id
    point = 0

    for x, stage in enumerate(PROCESS):
        if stage == game_copy.round:
            if stage in NONE_BET:
                game_copy.run(0, 0)
                game_copy.call = game_copy.minbet
                game_copy.max_bet = min(game_copy.players_pocket) - MIN_BET
            else:
                if phase == 1:
                    for i in range(game_copy.num_players):
                        num = (i + 1 + idx) % game_copy.num_players
                        check = False
                        # print('check_pt1 in sim')
                        while game_copy.unfold_list[num] and not check:
                            prob, input = agent.policy(num, game_copy)
                            bet, action = value_sampler(num, prob+multi*torch.rand(1, 64), game_copy)
                            check = game_copy.run(num, bet)
                        if sum(game_copy.unfold_list) == 1:
                            bet = ('check', game_copy.call)
                            check = game_copy.run(id, bet)
                            break
                        if num == id and check:
                            point += agent.value(input).item()
                        if num == button + 2:
                            phase = 2
                            idx = button + 2
                            break
                if phase == 2:
                    for i in range(game_copy.num_players):
                        num = (i + 1 + idx) % game_copy.num_players
                        check = False
                        while game_copy.unfold_list[num] and not check:
                            prob, input = agent.policy(num, game_copy)
                            bet, action = value_sampler(num, prob+multi*torch.rand(1, 64), game_copy)
                            if bet[0] == 'raise':
                                check = False
                            else:
                                check = game_copy.run(id, bet)
                        if sum(game_copy.unfold_list) == 1:
                            bet = ('check', game_copy.call)
                            check = game_copy.run(id, bet)
                            break
                        if num == id and check:
                            point += agent.value(input).item()
                        if num == button + 2:
                            phase = 1
                            idx = button + 2
                            break
            if game_copy.round == 'settle':
                win_lost = game_copy.players_pocket[id] / game_copy.players_origin[id] - 1
                point += ratio*win_lost
            else:
                game_copy.round = PROCESS[x+1]

    return point


def tree_search(args: tuple):
    handcomm, phase, agent, guessrecords, id, current_game, st = args
    print('\r','id:', id, current_game.round, 'phase:', phase, 'tree search', st+1, end='')
    players_copy = player_copier(current_game.players)
    players_copy[id].hand_comm = handcomm[:]
    game_copy = game_copier(id, current_game, players_copy)
    game_copy.deck.deck = handcomm[:]
    chip_ratio = [1 - game_copy.players_pocket[i] / game_copy.players_origin[i] for i in
                  range(game_copy.num_players)]
    prob, inputt = agent.policy(id, game_copy)
    bet, action = value_sampler(id, prob, game_copy)
    check = game_copy.run(id, bet)
    if game_copy.unfold_list[id] and not check:
        while game_copy.unfold_list[id] and not check:
            prob, inputt = agent.policy(id, game_copy)
            bet, action = value_sampler(id, prob, game_copy)
            check = game_copy.run(id, bet)

    ini_point = agent.value(inputt).item()

    for i in range(game_copy.num_players-1):
        # print(i)
        temp = torch.zeros(size=(1, 2))
        temp[0, 0] = chip_ratio[id - i - 1]
        temp[0, 1] = game_copy.unfold_list[id - i - 1]
        dispatched = 0
        for k in range(52):
            if game_copy.deck.deck[k] == 1:
                dispatched += 1
        input_m = torch.zeros(size=(1, 52))
        for g in range(52):
            if game_copy.deck.deck[g] == 1:
                if game_copy.community[g] == 1:
                    input_m[0, g] = 1
                elif current_game.players[id].hand[g] == 1:
                    input_m[0, g] = 0.5
                else:
                    input_m[0, g] = 0.1
        input1 = torch.cat((input_m, temp), 1)
        exp_out = agent.guesser(input1)
        # print(prob)
        for var in range(52):
            if game_copy.deck.deck[var] == 1:
                exp_out[0, var] = 0
        # print(exp_out)
        prob = exp_out / torch.sum(exp_out)
        # print('prob1:', prob)
        sample1 = torch.multinomial(prob, 1).item()
        exp_out[0, sample1] = 0
        prob = exp_out / torch.sum(exp_out)
        # print('prob2:', prob)
        sample2 = torch.multinomial(prob, 1).item()
        card = torch.zeros(size=(1, 52))
        card[0, sample1] = 1
        card[0, sample2] = 1
        game_copy.deck.deck[sample1] = 1
        game_copy.deck.deck[sample2] = 1
        if torch.rand(1).item() < 0.01:
            guessrecords.append((id-i-1, input1, dispatched))
        game_copy.players[id - i - 1].hand = card.tolist()[0]
        game_copy.players[id - i - 1].renew_handcomm(game_copy.community)
    game_copy.players[id].renew_handcomm(game_copy.community)
    # print('check_pt1:', t)
    point = sim(phase, agent, id, game_copy)
    # print('point')

    return point+ini_point, action, inputt


def monte_carlo(phase, agent, guessrecords: list, id: int, current_game: Game, search_time=2):
    points = [0 for _ in range(64)]
    handcomm = current_game.players[id].hand_comm
    actions = []

    # tree_search(handcomm, phase, agent, guessrecords, id, current_game, st)

    # pool = Pool(processes=Num_Processes)
    mapinput = []
    for xx in range(search_time):
        intuple = (handcomm, phase, agent, guessrecords, id, current_game, xx)
        mapinput.append(intuple)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(tree_search, mapinput)
    print('')

    for result in results:
        point, action, inputt = result
        points[action] += point
        actions.append(action)
            # print('check_pt2:', t, point)
    for cc in range(64):
        if cc not in actions:
            points[cc] = float('-inf')
    return points, inputt


def table(agent, guessrecord, card: list, recorder: list, stage, game: Game, type, st, multi=0.002):
    game.round = stage
    # print('round:', stage)
    button = game.button
    if stage in NONE_BET:
        game.run(0, 0)
        game.call = game.minbet
        game.round = stage
        # print(game.community)
        if stage == 'flop' or stage == 'turn' or stage == 'river':
            card.append(game.community[:])
    else:
        game.round = stage
        for i in range(game.num_players):
            phase = 1
            id = (i + button + 3) % game.num_players
            check = False
            action = None
            if type == 'mcts':
                points = [float('-inf')]
                # points, input = agent.mcts(phase, agent, guessrecord, id, game, max(2 ** st, 128))
                # print(points)
                while game.unfold_list[id] and not check:
                    if max(points) == float('-inf'):
                        while max(points) == float('-inf'):
                            # print(stage, 'mcts:', id)
                            points, input = agent.mcts(phase, agent, guessrecord, id, game, max(2 ** (st+2), 256))
                            # print('search done!', id)

                    bet, action, points = mcts_sample(id, points, game)
                    check = game.run(id, bet)
                    if sum(game.unfold_list) == 1:
                        bet = ('check', game.call)
                        check = game.run(id, bet)
                        break
                    if stage == 'pre-flop' and bet[0] == 'check':
                        check = False
            else:
                while game.unfold_list[id] and not check:
                    # print('what happened')
                    prob, input = agent.policy(id, game)
                    bet, action = value_sampler(id, prob+multi*torch.rand(1, 64), game)

                    check = game.run(id, bet)
                    # print(bet, check)
                    if sum(game.unfold_list) == 1:
                        bet = ('check', game.call)
                        check = game.run(id, bet)
                        break
                    if stage == 'pre-flop' and bet[0] == 'check':
                        check = False
            if action:
                recorder.append((id, action, input))
        for i in range(game.num_players):
            phase = 2
            # print(game.unfold_list, button)
            id = (i + button + 3) % game.num_players
            check = False
            action = None

            if type == 'mcts':
                points = [float('-inf')]
                while game.unfold_list[id] and not check:
                    # points, input = agent.mcts(phase, agent, guessrecord, id, game, max(2 ** st, 128))
                    if max(points) == float('-inf'):
                        while max(points) == float('-inf'):
                            # print(stage, 'mcts:', id)
                            points, input = agent.mcts(phase, agent, guessrecord, id, game, max(2 ** (st+2), 256))
                            # print('search done!', id)

                    bet, action, points = mcts_sample(id, points, game)
                    if bet[0] == 'raise':
                        check = False
                    else:
                        check = game.run(id, bet)
                    if sum(game.unfold_list) == 1:
                        bet = ('check', game.call)
                        check = game.run(id, bet)
                        break
                    if stage == 'pre-flop' and bet[0] == 'check':
                        check = False
            else:
                while game.unfold_list[id] and not check:
                    # print('what happened')
                    prob, input = agent.policy(id, game)
                    bet, action = value_sampler(id, prob+multi*torch.rand(1, 64), game)
                    # print(bet, check)
                    if bet[0] == 'raise':
                        check = False
                    else:
                        # print(id, game.round, bet, game.players[id].hc)
                        check = game.run(id, bet)
                    if sum(game.unfold_list) == 1:
                        bet = ('check', game.call)
                        check = game.run(id, bet)
                        break
                    if stage == 'pre-flop' and bet[0] == 'check':
                        check = False

            if action:
                recorder.append((id, action, input))


def decision_train(reward, recorder, policy, value, discount=0.8):
    policy.train()
    value.train()
    value_loss = 0
    policy_loss = 0
    size = 0
    for idx in range(len(recorder)):
        id, action, input = recorder[-idx-1]
        out = value.forward(input)
        td = out - reward[id]
        td = td.clone().detach()
        value_loss += td*out
        exp_out = policy(input)
        policy_loss += td*torch.log(exp_out[0, action] / torch.sum(exp_out))
        size += 1
        reward[id] *= discount

    return policy_loss / size, value_loss / size


def guess_train(game, guessrecord, guesser: CardGuesser):
    guesser.train()
    loss = 0
    size = 0

    for record in guessrecord:
        guessee_id, input, dispatched = record
        out = guesser.forward(input)
        # reward += card_reward(card.tolist()[0], hands[guessee_id]) / dispatched
        hands = game.players[guessee_id].hand
        hand1 = 0
        hand2 = 0
        for c in range(52):
            if hands[c] == 1:
                hand1 = c
            elif hands[c] == 1 and c != hand1:
                hand2 = c
        suits = [hand1 % 4, hand2 % 4]
        nums = [hand1 % 13, hand2 % 13]
        ten_suits1 = torch.zeros(size=(13, 4))
        ten_suits2 = torch.zeros(size=(13, 4))
        ten_nums1 = torch.zeros(size=(13, 4))
        ten_nums2 = torch.zeros(size=(13, 4))
        ten_suits1[:, suits[0]] = torch.ones(size=(13,))
        ten_suits2[:, suits[1]] = torch.ones(size=(13,))
        ten_nums1[nums[0], :] = torch.ones(size=(4,))
        ten_nums2[nums[1], :] = torch.ones(size=(4,))
        ten_suits1 = ten_suits1.reshape((1, 52))
        ten_suits2 = ten_suits2.reshape((1, 52))
        ten_nums1 = ten_nums1.reshape((1, 52))
        ten_nums2 = ten_nums2.reshape((1, 52))
        guess_out = out / torch.sum(out)
        loss += torch.matmul(torch.log(guess_out),
                             torch.transpose((ten_suits1 + ten_suits2) / 13 + (ten_nums1 + ten_nums2) / 4, 0,
                                             1))  # Simple Cross Entropy
        size += 1
    if size != 0:
        return loss / size
    else:
        return None


# def card_reward(pair1, pair2):
#     ppair1 = []
#     ppair2 = []
#     for i in range(52):
#         if pair1[i] == 1:
#             ppair1.append(i % 13)
#             ppair1.append(i % 4)
#         if pair2[i] == 1:
#             ppair2.append(i % 13)
#             ppair2.append(i % 4)
#     # print(ppair1, ppair2)
#     # print(pair1)
#     score1 = 1
#     if ppair1[0] - ppair2[0] == 0:
#         multi = 13
#     elif abs(ppair1[0] - ppair2[0]) == 1:
#         multi = 6
#     else:
#         multi = 1
#     score1 *= multi
#     if ppair1[1] - ppair2[1] == 0:
#         multi = 4
#     else:
#         multi = 1
#     score1 *= multi
#     if ppair1[2] - ppair2[2] == 0:
#         multi = 13
#     elif abs(ppair1[2] - ppair2[2]) == 1:
#         multi = 6
#     else:
#         multi = 1
#     score1 *= multi
#     if ppair1[3] - ppair2[3] == 0:
#         multi = 4
#     else:
#         multi = 1
#     score1 *= multi
#
#     score2 = 1
#     if ppair1[0] - ppair2[2] == 0:
#         multi = 13
#     elif abs(ppair1[0] - ppair2[2]) == 1:
#         multi = 6
#     else:
#         multi = 1
#     score2 *= multi
#     if ppair1[1] - ppair2[3] == 0:
#         multi = 4
#     else:
#         multi = 1
#     score2 *= multi
#     if ppair1[2] - ppair2[0] == 0:
#         multi = 13
#     elif abs(ppair1[2] - ppair2[0]) == 1:
#         multi = 6
#     else:
#         multi = 1
#     score2 *= multi
#     if ppair1[3] - ppair2[1] == 0:
#         multi = 4
#     else:
#         multi = 1
#     score2 *= multi
#
#     return max(score1, score2) - 2


def main(agent, policy, value, guesser, p_opt, v_opt, g_opt, button, money: list, player_num=6):
    deck = Deck()
    players = ()

    for i in range(player_num):
        player = Player(money[i])
        players += (player,)

    game = Game(players, deck)
    game.button = button
    step = 0
    recorder = []
    card = []
    guessrecord = []
    for st, stage in enumerate(PROCESS):
        # print(stage, step)
        # print(players[game.button].hc_show)
        if stage == 'settle':
            print('\n', 'who last', game.unfold_list)
        if step >= 4:
            table(agent, guessrecord, card, recorder, stage, game, 'mcts', st)
        else:
            table(agent, guessrecord, card, recorder, stage, game, 'policy', st)
        step += 1

    rest_money = [0 for _ in range(game.num_players)]
    for x in range(game.num_players):
        rest_money[x] = game.players_pocket[x]

    alist = rest_money[:]
    for c in range(game.num_players):
        win = alist.index(max(alist))
        rest_money[win] += 400 * c
        alist[win] = -1000

    id = 0
    reward = []
    hands = []
    for player in game.players:
        print('id:', id + 1, 'money:', game.players[id].money, game.players[id].hc_show)

        reward.append(game.players_pocket[id] / game.players_origin[id] - 1)
        hands.append(game.players[id].hand[:])
        print('best hand:', card_checker(player.hc))
        id += 1

    # print('community:', card)
    # print(guessrecord)
    # print(hands)



    policy_loss, value_loss = decision_train(reward, recorder, policy, value)
    guesser_loss = guess_train(game, guessrecord, guesser)

    p_opt.zero_grad()
    policy_loss.backward()
    p_opt.step()

    v_opt.zero_grad()
    value_loss.backward()
    v_opt.step()

    if guesser_loss is not None:
        g_opt.zero_grad()
        guesser_loss.backward()
        g_opt.step()

    return rest_money


if __name__ == '__main__':

    eras = 8

    for era in range(eras):
        policy = Policy()
        value = Value()
        guesser = CardGuesser()

        policy.load_state_dict(torch.load(pathp))
        value.load_state_dict(torch.load(pathv))
        guesser.load_state_dict(torch.load(pathg))

        learning_rate = 0.001
        p_opt = torch.optim.SGD(policy.parameters(), lr=learning_rate)
        v_opt = torch.optim.SGD(value.parameters(), lr=learning_rate)
        g_opt = torch.optim.SGD(guesser.parameters(), lr=learning_rate)

        epochs = 8

        number = torch.randint(4, 11, (1,)).tolist()[0]
        money = [1000 for _ in range(number)]
        ini_button = torch.randint(0, 6, size=()).item()
        for epoch in range(epochs):
            agent = Agent(policy, value, guesser)
            button = (epoch + ini_button) % number
            print('era: {}/{}, epoch: {}/{}'.format(era+1, eras, epoch+1, epochs))
            print('initial money:', money, 'button:', button)
            money = main(agent, policy, value, guesser, p_opt, v_opt, g_opt, button, money, player_num=number)

        torch.save(policy.state_dict(), pathp)
        torch.save(value.state_dict(), pathv)
        torch.save(guesser.state_dict(), pathg)






